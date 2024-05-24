# Video settings
VIDEO_FILES = [
    "videos/camera1.mp4",
    "videos/camera2.mp4",
]

# YOLO model settings
YOLO_MODEL_PATH = "yolov8x.pt"
CONFIDENCE_THRESHOLD = 0.5

# Detection settings
FRAME_SKIP = 5

# Coordinate system settings
COMMON_COORDINATE_SYSTEM_SCALE = 1000
COORDINATE_MATCH_THRESHOLD = 5

# Pygame GUI settings
GUI_SETTINGS = {
    "WINDOW_TITLE": "Multi-Camera Tracking System",
    "WINDOW_WIDTH": 800,
    "WINDOW_HEIGHT": 600,
    "FRAME_RATE": 60,
    "BACKGROUND_COLOR": (255, 255, 255),
    "CIRCLE_COLOR": (255, 0, 0),
    "CIRCLE_RADIUS": 3,
    "TEXT_COLOR": (0, 0, 255),
    "FONT_SIZE": 24
}

# Queue processing delay
QUEUE_PROCESS_DELAY = 0.05
