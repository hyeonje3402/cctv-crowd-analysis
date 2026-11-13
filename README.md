# CCTV 인파 밀집도 분석

녹화된 CCTV 영상에서 인원 수와 밀집도를 분석하고, 구역별 혼잡도를 JSON과 시각 자료로 확인하는 Python 컴퓨터 비전 프로젝트입니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 목적 | 행사장·공공장소 영상의 인파 규모와 혼잡 수준 분석 |
| 분석 방식 | CSRNet 밀도 맵 기반 인원 추정, YOLOv8 사람 객체 탐지 기반 배치 분석 |
| 시각화 | 분석 결과를 영상·그래픽으로 표현 |
| 기술 스택 | Python, PyTorch, Torchvision, OpenCV, Ultralytics YOLO, NumPy, SciPy, Pillow |
| 실행 형태 | 로컬 영상 파일을 처리하는 스크립트 |
| 저장·배포 | 결과 JSON을 로컬에 저장. 서비스 API, DB, 클라우드 배포 설정은 없음 |

## 분석 흐름

~~~mermaid
flowchart LR
    A[로컬 CCTV 영상] --> B{분석 스크립트 선택}
    B --> C[CSRNet 밀도 맵·인원 추정]
    B --> D[YOLOv8 사람 검출·시간 샘플링]
    C --> E[혼잡 단계와 구역별 JSON]
    D --> E
    E --> F[시각화 스크립트]
~~~

두 분석기는 서로 다른 모델과 혼잡도 임계값을 사용하므로 결과를 직접 같은 기준의 점수로 비교하지 않습니다.

## 상세 문서

- [분석 아키텍처와 알고리즘](docs/architecture/README.md)
- [설치·입력 영상·실행 방법](docs/development/README.md)
- [결과 해석과 개선 로드맵](docs/roadmap/README.md)

## 빠른 시작

~~~bash
pip install -r requirements.txt
python analyze_crowd.py
~~~

분석 대상 영상은 프로젝트 루트의 **videos/** 폴더에 준비합니다. 자세한 입력·출력 경로와 YOLO 배치 분석 명령은 [개발 문서](docs/development/README.md)를 참고하세요.