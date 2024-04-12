# 사격장 안전 확보 시스템

## 개요
본 프로젝트는 카메라를 통해 사수의 움직임을 관찰하고, 허가되지 않은 사격 징후나 안전 수칙 준수 여부 등의 이상 징후를 탐지하여 통제관, 부사수 등의 통제 요원에게 보고하는 시스템을 구축하는 것을 목표로 합니다.

## 주요 기능
- 허가되지 않은 사격 징후 및 안전 수칙 준수 여부 등의 이상 징후 탐지
- 행동 분석을 통한 잠재적 위험 상황 예측
- 탐지된 이상 징후를 통제관, 부사수 등의 통제 요원에게 보고
- 사용자 맞춤형 보고서 생성

## 사용 기술
- BoT-SORT
- NumPy
- OpenCV
- SlowFast
- Threading
- YOLOv8

## 파일 구조
```
📦SecureRange
 ┣ 📂src
 ┃ ┣ 📜config.py
 ┃ ┗ 📜main.py
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

## 설치 및 실행 방법
1. 저장소를 클론합니다.
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. 필요한 라이브러리를 설치합니다.
    ```bash
   pip install -r requirements.txt
   ```
3. config.py 파일에서 RTSP 스트림 URL과 기타 설정을 수정합니다.
4. 프로그램을 실행합니다.
    ```bash
   python main.py
   ```

# 라이선스
이 프로젝트는 AGPL-3.0 라이선스를 따릅니다.