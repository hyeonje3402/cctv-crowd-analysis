import os
import glob
import json
import urllib.request
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# ====================================================
# [설정 및 혼잡도 임계값]
# 유등축제 등 대규모 밀집 인파 기준 (CSRNet 밀도 맵 기반)
# ====================================================
CONGESTED_THRESHOLD = 300  # 300명 이상: 'CONGESTED' (위험/밀집)
CAUTION_THRESHOLD = 150    # 150명 이상: 'CAUTION' (주의)
                           # 150명 미만: 'NORMAL' (원활)

# 2944x1248 초고해상도 영상을 비율 유지하며 리사이징 (896x384: 속도 2.6초/프레임 및 높은 밀도 정확도)
TARGET_RESIZE_WIDTH = 896
TARGET_RESIZE_HEIGHT = 384

WEIGHTS_URL = "https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth"
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSRNet.pth")

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_FOLDER = os.path.join(WORKSPACE_DIR, "videos")
OUTPUT_JSON_FILE = os.path.join(WORKSPACE_DIR, "crowd_analysis.json")

# ====================================================
# [CCTV Zone ID 매핑 테이블]
# 파일명별 지정 zone_id가 없을 경우 자동으로 ZONE_01, ZONE_02... 형식으로 자동 부여됩니다.
# 나중에 특정 구역 이름(예: "ZONE_부교1", "CCTV_촉석루")으로 자유롭게 변경할 수 있습니다.
# ====================================================
ZONE_MAP = {
    # 예시:
    # "Hailuo_Video_Authentic CCTV footage, high a_567046910830706688.mp4": "ZONE_01",
}

def get_zone_id(video_name: str, index: int = 1) -> str:
    """비디오 파일명에 해당하는 zone_id를 반환합니다. 매핑이 없으면 ZONE_01 형식으로 자동 생성."""
    if video_name in ZONE_MAP:
        return ZONE_MAP[video_name]
    return f"ZONE_{index:02d}"



# ====================================================
# [CSRNet 아키텍처 정의]
# ====================================================
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        # VGG-16 기반 프론트엔드 (Conv 1_1 ~ Conv 4_3)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # Dilated Convolution 백엔드 (Dilation rate = 2)
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

def load_csrnet_model(device: torch.device):
    """CSRNet 모델 및 가중치를 로드합니다 (미보유 시 자동 다운로드)."""
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[*] CSRNet 가중치 파일 다운로드 중: {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print(f"[✔] 다운로드 완료: {WEIGHTS_PATH}")

    model = CSRNet()
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"[*] CSRNet 밀집 인파 분석 모델 로드 완료 (Device: {device})")
    return model

def determine_congestion(person_count: int) -> str:
    """
    유등축제 등 대규모 인파 환경 기준 혼잡도 판정:
      - 300명 이상: 'CONGESTED' (위험/밀집)
      - 150명 이상: 'CAUTION' (주의)
      - 150명 미만: 'NORMAL' (원활)
    """
    if person_count >= CONGESTED_THRESHOLD:
        return "CONGESTED"
    elif person_count >= CAUTION_THRESHOLD:
        return "CAUTION"
    else:
        return "NORMAL"

def analyze_video(video_path: str, model: CSRNet, device: torch.device, transform, zone_id: str = "ZONE_01") -> dict:
    """단일 비디오를 1초 간격으로 샘플링하여 CSRNet 밀도 맵으로 인원수를 추정합니다."""
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"[!] 비디오를 열 수 없습니다: {video_path}")
        return {
            "zone_id": zone_id,
            "file_name": video_name,
            "video_path": video_path,
            "error": "Failed to open video file",
            "frames": []
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 30.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2)
    step_frames = max(1, int(round(fps)))  # 1초 간격(fps 단위)

    print(f"\n==================================================")
    print(f"[▶] 영상 밀집도 분석 시작 (CSRNet): {video_name} (구역: {zone_id})")
    print(f"    - 원본 해상도: {orig_w}x{orig_h} -> 분석 해상도: {TARGET_RESIZE_WIDTH}x{TARGET_RESIZE_HEIGHT}")
    print(f"    - 총 프레임: {total_frames} | FPS: {fps:.2f} | 재생 시간: {duration_sec}초")
    print(f"    - 분석 간격: 매 {step_frames}프레임(약 1초마다)")
    print(f"--------------------------------------------------")

    frames_data = []
    frame_idx = 0
    analyzed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1초 간격으로만 모델 추론 실행
        if frame_idx % step_frames == 0:
            time_sec = round(frame_idx / fps, 2)

            # 1. 비율 유지 리사이즈 (와이드 비율)
            resized = cv2.resize(frame, (TARGET_RESIZE_WIDTH, TARGET_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
            # 2. RGB 변환 및 정규화 텐서 생성
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_tensor = transform(rgb).unsqueeze(0).to(device)

            # 3. CSRNet 밀도 맵(Density Map) 추론 및 인원수 적분(sum)
            with torch.no_grad():
                density_map = model(img_tensor)
                estimated_count = density_map.squeeze().sum().item()

            person_count = max(0, int(round(estimated_count)))
            status = determine_congestion(person_count)

            frame_entry = {
                "zone_id": zone_id,
                "file_name": video_name,
                "time_sec": time_sec,
                "timestamp": f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}",
                "frame_idx": frame_idx,
                "person_count": person_count,
                "raw_density_sum": round(estimated_count, 2),
                "status": status
            }
            frames_data.append(frame_entry)
            analyzed_count += 1

            print(f"  [{frame_entry['timestamp']} ({time_sec:5.2f}s) | Frame {frame_idx:4d}] "
                  f"추정 군중수: {person_count:4d}명 (DensitySum: {estimated_count:6.1f}) -> 혼잡도: {status}")

        frame_idx += 1

    cap.release()

    counts = [f["person_count"] for f in frames_data]
    max_count = max(counts) if counts else 0
    avg_count = round(sum(counts) / len(counts), 2) if counts else 0
    overall_status = determine_congestion(int(round(avg_count)))

    return {
        "zone_id": zone_id,
        "file_name": video_name,
        "video_path": os.path.relpath(video_path, WORKSPACE_DIR),
        "total_duration_sec": duration_sec,
        "total_frames_analyzed": analyzed_count,
        "max_person_count": max_count,
        "avg_person_count": avg_count,
        "overall_status": overall_status,
        "frames": frames_data
    }

def analyze_folder(folder_name: str, folder_path: str, model: CSRNet, device: torch.device, transform) -> list:
    """폴더 내의 모든 .mp4 영상을 CSRNet으로 분석합니다."""
    print(f"\n##################################################")
    print(f"[*] 폴더 탐색 시작: [{folder_name}] ({folder_path})")
    print(f"##################################################")

    if not os.path.exists(folder_path):
        print(f"[!] 폴더가 존재하지 않습니다: {folder_path}")
        return []

    video_files = glob.glob(os.path.join(folder_path, "*.mp4")) + \
                  glob.glob(os.path.join(folder_path, "*.MP4"))
    video_files = sorted(list(set(video_files)))

    print(f"[*] 탐지된 비디오 파일 개수: {len(video_files)}개")

    results = []
    for idx, video_file in enumerate(video_files, start=1):
        v_name = os.path.basename(video_file)
        zone_id = get_zone_id(v_name, index=idx)
        video_result = analyze_video(video_file, model, device, transform, zone_id=zone_id)
        results.append(video_result)

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_csrnet_model(device)

    # ImageNet 표준 정규화 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    analysis_results = analyze_folder("videos", VIDEOS_FOLDER, model, device, transform)

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print(f"\n[✔] videos 폴더 분석 완료 -> '{os.path.basename(OUTPUT_JSON_FILE)}' 저장 완료! (총 {len(analysis_results)}개 영상)")

    print(f"\n==================================================")
    print(f"[★] 모든 CSRNet 인파 분석 작업이 완료되었습니다.")
    print(f"==================================================")

if __name__ == "__main__":
    main()
