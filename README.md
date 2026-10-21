# CCTV Crowd Density Analysis System (캡스톤 디자인 인파 혼잡도 분석)

인파 밀집 구역의 안전 사고를 예방하기 위한 CCTV 영상 기반 실시간/배치 군중 밀도 분석 및 혼잡도 시각화 시스템입니다.

## 📌 주요 기능
1. **CSRNet 기반 고밀도 인파 추정 (`analyze_crowd.py`)**
   - 대규모 군중 밀집 지역(축제, 행사장 등)의 밀도 맵(Density Map) 생성 및 인원수 추정
   - 임계값 기반 혼잡도 단계 판정 (`NORMAL`, `CAUTION`, `CONGESTED`)
   - 구역별 JSON 리포트(`crowd_analysis.json`) 자동 생성

2. **YOLOv8 기반 배치 객체 감지 및 혼잡도 분석 (`batch_crowd_analysis.py`)**
   - YOLO 모델을 활용한 복수 CCTV 영상의 일괄(Batch) 인원 탐지
   - 구역(Zone)별 혼잡도 분석 결과 집계

3. **시각화 모듈 (`visualize_crowd.py`)**
   - 분석된 밀도 및 감지 결과를 영상 및 그래픽으로 시각화

---

## 🛠 설치 및 환경 설정

### 1. 가상환경 구성 및 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 모델 가중치 (Weights) 안내
- **CSRNet**: `analyze_crowd.py` 실행 시 Hugging Face에서 자동으로 다운로드되거나 필요 시 다운로드받아 프로젝트 루트에 배치합니다.
- **YOLOv8**: `ultralytics` 라이브러리를 통해 자동 다운로드됩니다 (`yolov8n.pt`, `yolov8s.pt` 등).

> **참고**: 대용량 모델 가중치(`.pt`, `.pth`) 및 영상 파일(`.mp4`)은 저장소 용량 관리를 위해 `.gitignore`에 포함되어 Git 추적에서 제외됩니다.

---

## 🚀 실행 방법

### CSRNet 인파 분석 실행
```bash
python analyze_crowd.py
```

### YOLOv8 배치 분석 실행
```bash
python batch_crowd_analysis.py
```

### 시각화 실행
```bash
python visualize_crowd.py
```
