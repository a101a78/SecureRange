import numpy as np

# Video settings
VIDEO_FILES = [
    'videos/camera1.mp4',
    'videos/camera2.mp4',
    'videos/camera3.mp4',
]  # Paths to video files

# YOLO model settings
YOLO_MODEL_PATH = 'yolov8x.pt'  # Path to the YOLO model file
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold for object detection

# Detection settings
FRAME_SKIP = 5  # Number of frames to skip between each detection
OBJECT_TIMEOUT = 1000  # Maximum time (in milliseconds) for an object to be considered lost
MAX_DISTANCE_THRESHOLD = 50  # Maximum distance (in pixels) to consider two objects as the same

# GUI settings
GUI_SETTINGS = {
    'WINDOW_TITLE': 'Multi-Camera Tracking System',  # Title of the GUI window
    'WINDOW_WIDTH': 800,  # Width of the GUI window in pixels
    'WINDOW_HEIGHT': 600,  # Height of the GUI window in pixels
    'FRAME_RATE': 60,  # Target frame rate for the GUI update loop
    'BACKGROUND_COLOR': (255, 255, 255),  # Background color
    'CIRCLE_COLOR': (255, 0, 0),  # Color of the circles representing objects
    'CIRCLE_RADIUS': 3,  # Radius of the circles in pixels
    'TEXT_COLOR': (0, 0, 255),  # Color of the text displaying object coordinates
    'FONT_SIZE': 24  # Font size for the text
}

# Queue processing delay
QUEUE_PROCESS_DELAY = 0.05  # Delay between each queue processing cycle in seconds

# Triangulation settings
TRIANGULATION_SETTINGS = {
    'SCALE_FACTOR': None,  # Scale factor (automatically calculated if None)
    'FIELD_WIDTH': 3.2,  # Width of the field (in meters)
    'FIELD_HEIGHT': 8.56,  # Height of the field (in meters)
    'CAMERA_POSITIONS': [  # Camera positions (x, y, z coordinates in meters)
        (0, 0, 1.7),
        (3.2, 0, 1.7),
        (3.2, 8.56, 1.7),
    ]
}

# Calculate SCALE_FACTOR automatically if not provided
if TRIANGULATION_SETTINGS['SCALE_FACTOR'] is None:
    field_area = TRIANGULATION_SETTINGS['FIELD_WIDTH'] * TRIANGULATION_SETTINGS['FIELD_HEIGHT']
    gui_area = GUI_SETTINGS['WINDOW_WIDTH'] * GUI_SETTINGS['WINDOW_HEIGHT']
    TRIANGULATION_SETTINGS['SCALE_FACTOR'] = np.sqrt(gui_area / field_area)
