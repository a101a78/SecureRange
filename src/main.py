import math
import threading
import time
from queue import Queue

import cv2
import pygame
from ultralytics import YOLO

from src import config
from utils.logger import AsyncLogger


class VideoProcessor(threading.Thread):
    """
    Thread class for processing video frames and detecting objects using YOLO model.

    Args:
        video_path (str): Path to the video file.
        queue (Queue): Queue to store detected bounding boxes.
        camera_id (int): ID of the camera (video file).
        logger (AsyncLogger): Logger instance for logging messages.

    Methods:
        run: Starts the video processing thread.
        stop: Stops the video processing thread.
    """

    def __init__(self, video_path, queue, camera_id, logger):
        super().__init__()
        self.video_path = video_path
        self.queue = queue
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.camera_id = camera_id
        self.stop_event = threading.Event()
        self.logger = logger

    def run(self):
        """
        Starts processing the video and detecting objects. Detected objects are placed in the queue.

        Raises:
            RuntimeError: If a frame cannot be read from the video file.
        """
        self.logger.log(f'Starting video processing for camera {self.camera_id}')
        while not self.stop_event.is_set() and self.cap.isOpened():
            for _ in range(config.FRAME_SKIP):
                self.cap.read()
            success, frame = self.cap.read()
            if not success:
                self.logger.log(f'Failed to read frame for camera {self.camera_id}')
                break
            results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD)
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                self.queue.put((self.camera_id, x1, y1, x2, y2))
        self.logger.log(f'Stopping video processing for camera {self.camera_id}')
        self.cap.release()

    def stop(self):
        """
        Stops the video processing thread.
        """
        self.stop_event.set()


class CommonCoordinateSystem:
    """
    Class to manage the common coordinate system for multi-camera object tracking.

    Args:
        logger (AsyncLogger): Logger instance for logging messages.

    Methods:
        update: Updates the coordinate system with detected object coordinates from a camera.
        get_triangulated_coordinates: Gets the triangulated coordinates from multiple cameras.
        triangulate: Static method to calculate the real-world coordinates using triangulation.
    """

    def __init__(self, logger):
        self.objects = {}
        self.next_id = 0
        self.lock = threading.Lock()
        self.logger = logger

    def update(self, camera_id, x1, y1, x2, y2):
        """
        Updates the common coordinate system with the detected object coordinates from a camera.

        Args:
            camera_id (int): ID of the camera.
            x1 (float): Top-left x-coordinate of the bounding box.
            y1 (float): Top-left y-coordinate of the bounding box.
            x2 (float): Bottom-right x-coordinate of the bounding box.
            y2 (float): Bottom-right y-coordinate of the bounding box.
        """
        with self.lock:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if camera_id not in self.objects:
                self.objects[camera_id] = []

            self.objects[camera_id].append((center_x, center_y))

    def get_triangulated_coordinates(self):
        """
        Get the triangulated coordinates from multiple cameras.

        Returns:
            list: List of tuples representing the triangulated coordinates (x, y, z).
        """
        triangulated_coords = []
        if len(self.objects) >= 2:
            keys = list(self.objects.keys())
            for idx1 in range(len(keys)):
                for idx2 in range(idx1 + 1, len(keys)):
                    cam1 = self.objects[keys[idx1]]
                    cam2 = self.objects[keys[idx2]]
                    camera_distance = self.calculate_camera_distance(keys[idx1], keys[idx2])
                    for (x1, y1), (x2, _) in zip(cam1, cam2):
                        try:
                            triangulated_coord = self.triangulate(x1, y1, x2, camera_distance)
                            triangulated_coords.append(triangulated_coord)
                        except RuntimeError as e:
                            self.logger.log(f'Error during triangulation: {e}')
        return triangulated_coords

    @staticmethod
    def triangulate(x1, y1, x2, camera_distance):
        """
        Calculate the real-world coordinates using triangulation.

        Args:
            x1 (float): x-coordinate from the first camera.
            y1 (float): y-coordinate from the first camera.
            x2 (float): x-coordinate from the second camera.
            camera_distance (float): Distance between the two cameras.

        Returns:
            tuple: Real-world coordinates (x, y, z).

        Raises:
            RuntimeError: If the x-coordinates from both cameras are identical.

        The roles of the coordinates are as follows:
            - x: Represents the horizontal position of the object.
            - y: Represents the vertical position of the object.
            - z: Represents the distance from the camera to the object (depth).
        """
        dx = x1 - x2
        if dx == 0:
            raise RuntimeError(f'Identical x-coordinates for triangulation: x1={x1}, x2={x2}')
        z = camera_distance / abs(dx)
        x = x1 * z
        y = y1 * z
        return x, y, z

    @staticmethod
    def calculate_camera_distance(cam1_id, cam2_id):
        """
        Calculate the distance between two cameras based on their IDs.

        Args:
            cam1_id (int): ID of the first camera.
            cam2_id (int): ID of the second camera.

        Returns:
            float: Distance between the two cameras.

        Raises:
            ValueError: If the camera pair is unknown.
        """
        cam_pair = {cam1_id, cam2_id}
        if cam_pair == {0, 1} or cam_pair == {0, 2}:
            return math.sqrt((config.TRIANGULATION_SETTINGS['FIELD_WIDTH'] / 2) ** 2 + config.TRIANGULATION_SETTINGS[
                'FIELD_HEIGHT'] ** 2)
        elif cam_pair == {1, 2}:
            return config.TRIANGULATION_SETTINGS['FIELD_WIDTH']
        else:
            raise ValueError(f'Unknown camera pair: {cam1_id}, {cam2_id}')


class GUI:
    """
    Class to manage the graphical user interface for displaying tracked objects.

    Args:
        common_coord_system (CommonCoordinateSystem): Instance of the CommonCoordinateSystem class.
        logger (AsyncLogger): Logger instance for logging messages.

    Methods:
        update_gui: Updates the GUI with the latest tracked object coordinates.
        run: Runs the GUI event loop.
    """

    def __init__(self, common_coord_system, logger):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.GUI_SETTINGS["WINDOW_WIDTH"], config.GUI_SETTINGS["WINDOW_HEIGHT"]))
        pygame.display.set_caption(config.GUI_SETTINGS["WINDOW_TITLE"])
        self.clock = pygame.time.Clock()
        self.common_coord_system = common_coord_system
        self.logger = logger

    def update_gui(self):
        """
        Updates the GUI with the latest tracked object coordinates.
        """
        self.screen.fill(config.GUI_SETTINGS["BACKGROUND_COLOR"])
        triangulated_coords = self.common_coord_system.get_triangulated_coordinates()
        for coord in triangulated_coords:
            if coord is not None:
                x, y, z = coord
                screen_x = int(x / z * config.TRIANGULATION_SETTINGS['SCALE_FACTOR'])
                screen_y = int(y / z * config.TRIANGULATION_SETTINGS['SCALE_FACTOR'])
                pygame.draw.circle(self.screen, config.GUI_SETTINGS["CIRCLE_COLOR"], (screen_x, screen_y),
                                   config.GUI_SETTINGS["CIRCLE_RADIUS"])
        pygame.display.flip()

    def run(self):
        """
        Runs the GUI event loop.

        Raises:
            pygame.error: If an error occurs in the Pygame event loop.
        """
        self.logger.log('Starting GUI')
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.update_gui()
            self.clock.tick(config.GUI_SETTINGS["FRAME_RATE"])
        self.logger.log('Stopping GUI')
        pygame.quit()


def main():
    with AsyncLogger() as logger:
        try:
            logger.log('Starting main function')

            common_coord_system = CommonCoordinateSystem(logger)
            gui = GUI(common_coord_system, logger)

            queues = [Queue() for _ in config.VIDEO_FILES]
            video_processors = [VideoProcessor(video_path, queues[i], i, logger) for i, video_path in
                                enumerate(config.VIDEO_FILES)]

            for vp in video_processors:
                vp.start()

            stop_event = threading.Event()

            def process_queue():
                logger.log('Starting queue processing')
                while not stop_event.is_set():
                    for q in queues:
                        while not q.empty():
                            camera_id, x1, y1, x2, y2 = q.get()
                            common_coord_system.update(camera_id, x1, y1, x2, y2)
                    time.sleep(config.QUEUE_PROCESS_DELAY)
                logger.log('Stopping queue processing')

            queue_thread = threading.Thread(target=process_queue)
            queue_thread.daemon = True
            queue_thread.start()

            gui.run()

            stop_event.set()
            for vp in video_processors:
                vp.stop()
                vp.join()

            queue_thread.join()

            logger.log('Stopping main function')
        except Exception as e:
            logger.log(f'Unhandled exception: {e}')


if __name__ == "__main__":
    main()
