import os
import glob
import json
import cv2
from ultralytics import YOLO

# ----------------------------------------------------
# 1. 설정 및 파라미터
# ----------------------------------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_FOLDER = os.path.join(WORKSPACE_DIR, "videos")
OUTPUT_JSON_FILE = os.path.join(WORKSPACE_DIR, "yolo_crowd_analysis.json")

# ====================================================
# [CCTV Zone ID 매핑 테이블]
# 파일명별 지정 zone_id가 없을 경우 자동으로 ZONE_01, ZONE_02... 형식으로 자동 부여됩니다.
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


# 로컬 캐시 모델 경로 탐색 (우선순위)
CANDIDATE_MODEL_PATHS = [
    os.path.join(WORKSPACE_DIR, "yolov8n.pt"),
    r"C:\Users\82107\.gemini\antigravity-ide\scratch\computer-vision\yolov8n.pt",
    r"C:\Users\82107\.gemini\antigravity-ide\scratch\computer-vision\yolo11n.pt",
    "yolov8n.pt"
]

def load_yolo_model():
    """사용 가능한 YOLO 모델을 로드합니다."""
    model_path = "yolov8n.pt"
    for candidate in CANDIDATE_MODEL_PATHS:
        if os.path.exists(candidate):
            model_path = candidate
            break
    print(f"[*] YOLO 모델 로드 중: {model_path}")
    return YOLO(model_path)

def determine_congestion(person_count: int) -> str:
    """
    인원수에 따른 혼잡도 판정 규칙:
      - 50명 이상: 'CONGESTED'
      - 25명 이상: 'CAUTION'
      - 그 미만: 'NORMAL'
    """
    if person_count >= 50:
        return "CONGESTED"
    elif person_count >= 25:
        return "CAUTION"
    else:
        return "NORMAL"

def analyze_video(video_path: str, model: YOLO, zone_id: str = "ZONE_01") -> dict:
    """단일 비디오를 1초 간격으로 샘플링하여 혼잡도를 분석합니다."""
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    if not cap.isOpened():
        print(f"[!] 비디오를 열 수 없습니다: {video_path}")
        return {
            "zone_id": zone_id,
            "video_name": video_name,
            "video_path": video_path,
            "error": "Failed to open video file",
            "frames": []
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2)
    step_frames = max(1, int(round(fps)))  # 1초 간격 프레임 수

    video_name = os.path.basename(video_path)
    print(f"\n==================================================")
    print(f"[▶] 영상 분석 시작: {video_name} (구역: {zone_id})")
    print(f"    - 총 프레임: {total_frames} | FPS: {fps:.2f} | 재생 시간: {duration_sec}초")
    print(f"    - 분석 간격: 매 {step_frames}프레임(약 1초)마다")
    print(f"--------------------------------------------------")

    frames_data = []
    frame_idx = 0
    analyzed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step_frames == 0:
            time_sec = round(frame_idx / fps, 2)

            # YOLO 사람(person, class 0) 탐지
            # classes=[0] 지정 시 사람 클래스만 필터링하여 추론
            results = model(frame, classes=[0], conf=0.25, verbose=False)
            boxes = results[0].boxes
            person_count = len(boxes) if boxes is not None else 0

            status = determine_congestion(person_count)

            frame_entry = {
                "zone_id": zone_id,
                "file_name": video_name,
                "time_sec": time_sec,
                "frame_idx": frame_idx,
                "person_count": person_count,
                "status": status
            }
            frames_data.append(frame_entry)
            analyzed_count += 1

            print(f"  [{time_sec:5.2f}s | Frame {frame_idx:4d}] 인원수: {person_count:3d}명 -> 혼잡도: {status}")

        frame_idx += 1

    cap.release()

    # 요약 통계 계산
    counts = [f["person_count"] for f in frames_data]
    max_count = max(counts) if counts else 0
    avg_count = round(sum(counts) / len(counts), 2) if counts else 0
    overall_status = determine_congestion(int(round(avg_count)))

    return {
        "zone_id": zone_id,
        "video_name": video_name,
        "file_name": video_name,
        "video_path": os.path.relpath(video_path, WORKSPACE_DIR),
        "total_duration_sec": duration_sec,
        "total_frames_analyzed": analyzed_count,
        "max_person_count": max_count,
        "avg_person_count": avg_count,
        "overall_status": overall_status,
        "frames": frames_data
    }

def analyze_folder(folder_name: str, folder_path: str, model: YOLO) -> list:
    """폴더 내의 모든 .mp4 영상을 분석합니다."""
    print(f"\n##################################################")
    print(f"[*] 폴더 탐색 시작: [{folder_name}] ({folder_path})")
    print(f"##################################################")

    if not os.path.exists(folder_path):
        print(f"[!] 폴더가 존재하지 않습니다: {folder_path}")
        return []

    # 대소문자 구분 없이 .mp4 파일 탐색
    video_files = glob.glob(os.path.join(folder_path, "*.mp4")) + \
                  glob.glob(os.path.join(folder_path, "*.MP4"))
    video_files = sorted(list(set(video_files)))

    print(f"[*] 탐지된 비디오 파일 개수: {len(video_files)}개")

    results = []
    for idx, video_file in enumerate(video_files, start=1):
        v_name = os.path.basename(video_file)
        zone_id = get_zone_id(v_name, index=idx)
        video_result = analyze_video(video_file, model, zone_id=zone_id)
        results.append(video_result)

    return results

def main():
    model = load_yolo_model()

    analysis_results = analyze_folder("videos", VIDEOS_FOLDER, model)

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)

    print(f"\n[✔] videos 폴더 분석 완료 -> '{os.path.basename(OUTPUT_JSON_FILE)}' 저장 완료! (총 {len(analysis_results)}개 영상)")

    print(f"\n==================================================")
    print(f"[★] 모든 일괄 분석 작업이 성공적으로 완료되었습니다.")
    print(f"==================================================")

if __name__ == "__main__":
    main()
