import os
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# ====================================================
# [설정 및 혼잡도 임계값]
# ====================================================
CONGESTED_THRESHOLD = 300  # 300명 이상: 혼잡 (CONGESTED)
CAUTION_THRESHOLD = 150    # 150명 이상: 주의 (CAUTION)
                           # 150명 미만: 원활 (NORMAL)

# 출력 및 분석 해상도 (2944x1248 원본 비율 약 2.35:1 유지)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 544

# 모델 가중치 경로
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, "CSRNet.pth")
VIDEOS_FOLDER = os.path.join(WORKSPACE_DIR, "videos")
OUTPUT_VIDEO_PATH = os.path.join(WORKSPACE_DIR, "crowd_result_video.mp4")


# 한글 폰트 로드 (Windows 맑은 고딕)
FONT_PATH = r"C:\Windows\Fonts\malgunbd.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"

def get_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

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
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

def load_csrnet(device: torch.device):
    """CSRNet 모델 로드"""
    model = CSRNet()
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"CSRNet.pth 가중치 파일이 없습니다: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"[*] CSRNet 모델 로드 완료 (Device: {device})")
    return model

def determine_congestion(person_count: float):
    if person_count >= CONGESTED_THRESHOLD:
        return "혼잡 (CONGESTED)", (40, 40, 225)       # 빨강 (BGR)
    elif person_count >= CAUTION_THRESHOLD:
        return "주의 (CAUTION)", (0, 140, 255)         # 주황 (BGR)
    else:
        return "원활 (NORMAL)", (60, 180, 75)          # 녹색 (BGR)

def apply_density_heatmap(frame_bgr: np.ndarray, density_map: np.ndarray) -> np.ndarray:
    """
    CSRNet 밀도 맵을 JET 컬러맵(파랑->녹색->노랑->빨강)으로 변환 후
    사람이 있는 영역에만 반투명하게 합성합니다.
    """
    h, w, _ = frame_bgr.shape
    # 1. 밀도 맵을 프레임 해상도로 확대
    density_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_CUBIC)
    density_resized = np.maximum(density_resized, 0)

    # 2. 정규화 (가시성 향상을 위해 동적 범위 스케일링)
    max_val = np.percentile(density_resized, 99.5) if density_resized.max() > 0 else 1.0
    if max_val <= 0:
        max_val = 1.0
    density_norm = np.clip(density_resized / max_val * 255.0, 0, 255).astype(np.uint8)

    # 3. JET 컬러맵 적용 (파랑 -> 녹색 -> 노랑 -> 빨강)
    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

    # 4. 배경 마스킹: 사람이 없는 빈 배경(물, 하늘 등)은 원본 유지
    # 밀도 임계값 이상인 곳에만 부드러운 가중치로 히트맵 블렌딩
    mask = (density_norm > 12).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)[:, :, np.newaxis]

    alpha = 0.55 * mask
    blended = (frame_bgr * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)
    return blended

def draw_hud_overlay(frame_bgr: np.ndarray, current_count: float, max_count: float, avg_count: float, fps_val: float) -> np.ndarray:
    """
    친구 영상 스크린샷과 100% 동일한 상단/하단 HUD 바를 Pillow로 선명하게 렌더링합니다.
    """
    h, w, _ = frame_bgr.shape
    status_label, badge_color_bgr = determine_congestion(current_count)

    # 1. 상단 블랙 바 (반투명)
    top_bar_height = 54
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_height), (12, 12, 16), -1)
    
    # 2. 하단 정보 바 (반투명)
    bot_bar_height = 30
    cv2.rectangle(overlay, (0, h - bot_bar_height), (w, h), (10, 10, 14), -1)
    cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0, frame_bgr)

    # 3. OpenCV(BGR) -> PIL(RGB) 변환하여 고품질 한글 텍스트 렌더링
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_main = get_font(18)
    font_badge = get_font(16)
    font_sub = get_font(14)

    # [상단 좌측 텍스트]
    hud_left_text = f"CSRNet 추정 인원: {current_count:.1f}명 (약 {int(round(current_count))}명) | 최대: {int(round(max_count))}명 | 평균: {avg_count:.1f}명"
    draw.text((20, 16), hud_left_text, font=font_main, fill=(255, 255, 255))

    # [상단 중앙 밀집 등급 배지]
    badge_rgb = (badge_color_bgr[2], badge_color_bgr[1], badge_color_bgr[0])
    badge_x = int(w * 0.58)
    badge_y = 10
    badge_w = 210
    badge_h = 34
    draw.rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], fill=badge_rgb, outline=(255, 255, 255), width=1)
    badge_text = f"밀집 등급: {status_label}"
    draw.text((badge_x + 12, badge_y + 6), badge_text, font=font_badge, fill=(255, 255, 255))

    # [상단 우측 모델 정보]
    hud_right_text = f"FPS: {fps_val:.1f} | 모델: CSRNet (SHB, 640p)"
    draw.text((badge_x + badge_w + 20, 18), hud_right_text, font=font_sub, fill=(190, 190, 190))

    # [하단 자막 텍스트]
    footer_text = "[V] 보기모드: 반투명 밀도 오버레이 | [C] 컬러맵: JET (레인보우) | [O] 원본 | 방식: 순수 밀도 추정 (End-to-End Regression)"
    draw.text((20, h - 23), footer_text, font=font_sub, fill=(180, 180, 180))

    # PIL -> OpenCV(BGR) 복원
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def process_folder(folder_key: str, folder_path: str, output_path: str, model: CSRNet, device: torch.device, transform):
    """지정된 폴더의 영상들을 CSRNet으로 시각화하여 종합 비디오로 렌더링합니다."""
    print(f"\n##################################################")
    print(f"[*] [{folder_key}] CSRNet 밀도 히트맵 동영상 생성 시작")
    print(f"    - 대상 폴더: {folder_path}")
    print(f"    - 출력 파일: {output_path}")
    print(f"##################################################")

    if not os.path.exists(folder_path):
        print(f"[!] 폴더가 없습니다: {folder_path}")
        return

    video_files = glob.glob(os.path.join(folder_path, "*.mp4")) + \
                  glob.glob(os.path.join(folder_path, "*.MP4"))
    video_files = sorted(list(set(video_files)))

    if not video_files:
        print(f"[!] {folder_key} 폴더에 영상 파일이 없습니다.")
        return

    # 첫 영상 기준 FPS
    probe_cap = cv2.VideoCapture(video_files[0])
    target_fps = probe_cap.get(cv2.CAP_PROP_FPS) or 24.0
    probe_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    all_counts = []
    total_frames = 0

    for v_idx, v_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(v_path)
        v_name = os.path.basename(v_path)
        v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n[▶] ({v_idx}/{len(video_files)}) 렌더링 중: {v_name} (총 {v_frames}프레임)")

        frame_idx = 0
        last_density = None
        last_count = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()
            # 규격 해상도로 리사이즈
            resized = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)

            # 매 6프레임마다 CSRNet 밀도 맵 추론 (연산 속도 최적화 및 부드러운 히트맵 유지)
            if frame_idx % 6 == 0 or last_density is None:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img_tensor = transform(rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    d_map = model(img_tensor)
                    last_density = d_map.squeeze().cpu().numpy()
                    last_count = float(last_density.sum())
                    all_counts.append(last_count)

            # 히트맵 합성
            heat_frame = apply_density_heatmap(resized, last_density)

            # 통계값 계산
            max_c = max(all_counts) if all_counts else last_count
            avg_c = sum(all_counts) / len(all_counts) if all_counts else last_count

            # 프레임당 추론 FPS 계산
            dt = time.time() - t_start
            inst_fps = 1.0 / dt if dt > 0 else target_fps

            # HUD 바 렌더링 (스크린샷과 동일한 디자인)
            final_frame = draw_hud_overlay(heat_frame, last_count, max_c, avg_c, inst_fps)

            writer.write(final_frame)
            frame_idx += 1
            total_frames += 1

            if frame_idx % 30 == 0 or frame_idx == v_frames:
                print(f"    - 진행률: {frame_idx:4d}/{v_frames} | 추정 인원: {last_count:5.1f}명")

        cap.release()

    writer.release()
    print(f"\n[✔] {folder_key} CSRNet 시각화 동영상 저장 완료! ({total_frames}프레임)")
    print(f"    -> {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_csrnet(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    process_folder("videos", VIDEOS_FOLDER, OUTPUT_VIDEO_PATH, model, device, transform)

    print(f"\n==================================================")
    print(f"[★] CSRNet 밀도 히트맵 시각화 통합 영상 생성이 완료되었습니다.")
    print(f"    -> {OUTPUT_VIDEO_PATH}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
