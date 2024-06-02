# Video settings
VIDEO_FILES = [
    "videos/camera1-1.mp4",
    "videos/camera1-2.mp4",
]

# YOLO model settings
YOLO_MODEL_PATH = "yolov8x.pt"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold for object detection

# Feature matching settings
FEATURE_MATCH_THRESHOLD = 0.5  # Threshold for matching features; higher values mean stricter matching
COST_THRESHOLD = 1.0  # Maximum allowed cost for matching objects between frames

# Detection settings
FRAME_SKIP = 5  # Number of frames to skip between each detection

# Coordinate system settings
COMMON_COORDINATE_SYSTEM_SCALE = 1000  # Scale factor for converting to a common coordinate system
COORDINATE_MATCH_THRESHOLD = 5  # Threshold for matching coordinates between frames

# Pygame GUI settings
GUI_SETTINGS = {
    "WINDOW_TITLE": "Multi-Camera Tracking System",
    "WINDOW_WIDTH": 800,
    "WINDOW_HEIGHT": 600,
    "FRAME_RATE": 60,  # Frame rate for the GUI update loop
    "BACKGROUND_COLOR": (255, 255, 255),
    "CIRCLE_COLOR": (255, 0, 0),
    "CIRCLE_RADIUS": 3,
    "TEXT_COLOR": (0, 0, 255),
    "FONT_SIZE": 24
}

# Queue processing delay
QUEUE_PROCESS_DELAY = 0.05  # Delay between each queue processing cycle in seconds
