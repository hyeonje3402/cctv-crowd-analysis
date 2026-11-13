# 설치와 실행

[메인 README](../../README.md)

## 준비물

Python 환경과 영상 파일이 필요합니다. PyTorch가 설치되는 CPU/GPU 환경에 맞춰 설치를 진행하세요.

~~~bash
git clone https://github.com/hyeonje3402/cctv-crowd-analysis.git
cd cctv-crowd-analysis
python -m venv .venv
~~~

Windows PowerShell에서 가상환경을 활성화한 뒤 의존성을 설치합니다.

~~~powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

## 입력 영상

스크립트는 저장소 루트 아래의 videos/ 디렉터리를 입력 위치로 사용합니다. 해당 폴더를 만들고 분석할 동영상을 넣으세요. 큰 영상 및 모델 가중치는 Git 저장소에서 제외되도록 설정되어 있을 수 있습니다.

## 실행

~~~bash
# CSRNet 밀도 맵 기반 인원 추정
python analyze_crowd.py

# YOLO 사람 검출 기반 배치 분석
python batch_crowd_analysis.py

# JSON 결과 시각화
python visualize_crowd.py
~~~

CSRNet 가중치는 analyze_crowd.py에서 없으면 다운로드합니다. YOLO 모델은 로컬 후보 경로를 확인한 뒤 Ultralytics 모델 경로를 사용합니다. 현재 batch_crowd_analysis.py에는 특정 Windows 개발 경로 후보도 남아 있으므로 다른 PC에서 실행할 때 모델 경로를 확인해야 합니다.

## 결과

CSRNet 결과는 crowd_analysis.json, YOLO 배치 결과는 yolo_crowd_analysis.json에 기록하도록 구성되어 있습니다. 영상이 열리지 않으면 파일 경로·코덱·OpenCV 설치를 확인하세요.

## 테스트

저장소에 자동 테스트나 CI 워크플로는 확인되지 않았습니다. 입력 영상의 구역별 정답 인원 수를 확보해 혼잡도 추정 오차를 검증하는 과정은 별도로 필요합니다.