# RTSP 스트림 URL 리스트
RTSP_URLS = [
    "rtsp://admin:password@192.168.0.100:554/stream1",
    "rtsp://admin:password@192.168.0.101:554/stream2",
    "rtsp://admin:password@192.168.0.102:554/stream3",
    "rtsp://admin:password@192.168.0.103:554/stream4"
]

# YOLOv8 모델 경로
YOLO_MODEL_PATH = 'yolov8n.pt'

# 프레임 읽기 재시도 관련 설정
MAX_RETRIES = 10  # 최대 재시도 횟수
RETRY_DELAY = 1  # 재시도 간격 (초)
