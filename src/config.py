# List of video files for different camera angles
VIDEO_FILES = ["videos/camera1.mp4", "videos/camera2.mp4", "videos/camera3.mp4"]

# Path to the YOLOv8 model weights file
YOLO_MODEL_PATH = "yolov8x.pt"

# Confidence threshold for YOLO object detection
# Objects with a detection confidence lower than this value will be ignored
CONFIDENCE_THRESHOLD = 0.5

# Number of frames to skip between each detection
# Higher values result in fewer detections but lower processing load
FRAME_SKIP = 5

# Scale factor for converting pixel coordinates to the common 2D coordinate system
# Larger values will result in smaller common coordinates
COMMON_COORDINATE_SYSTEM_SCALE = 1000

# Time duration in seconds for which the trajectory tail should be visible
# The tail represents the movement trajectory of the detected objects
TRAJECTORY_DWELL_TIME = 0.3

# Distance threshold for matching objects in the common coordinate system
# Objects within this distance are considered the same entity
COORDINATE_MATCH_THRESHOLD = 5
