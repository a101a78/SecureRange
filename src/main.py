import threading
from queue import Queue

import cv2
import numpy as np
import pygame
from ultralytics import YOLO

from src import config
from utils.logger import AsyncLogger


class VideoProcessor(threading.Thread):
    """Thread for processing video frames and detecting objects using YOLO."""

    def __init__(self, video_path, queue, camera_id, logger):
        """
        Initializes the VideoProcessor.

        Args:
            video_path (str): Path to the video file.
            queue (Queue): Queue to store detected bounding boxes.
            camera_id (int): ID of the camera (video file).
            logger (AsyncLogger): Logger instance for logging messages.
        """
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
        Processes the video, detects objects, and places their coordinates in the queue.
        """
        self.logger.log(f'Starting video processing for camera {self.camera_id}')
        while not self.stop_event.is_set() and self.cap.isOpened():
            for _ in range(config.FRAME_SKIP):
                self.cap.read()
            success, frame = self.cap.read()
            if not success:
                self.logger.log(f'Failed to read frame for camera {self.camera_id}')
                break

            results = self.model(frame, device=0, classes=0, conf=config.CONFIDENCE_THRESHOLD)
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                self.queue.put((self.camera_id, x1, y1, x2, y2))

        self.logger.log(f'Stopping video processing for camera {self.camera_id}')
        self.cap.release()

    def stop(self):
        """Stops the video processing thread."""
        self.stop_event.set()


class CommonCoordinateSystem:
    """
    Manages the common coordinate system for multi-camera object tracking.
    """

    def __init__(self, logger):
        """
        Initializes the common coordinate system.

        Args:
            logger (AsyncLogger): The logger object for logging messages.
        """
        self.objects = {}  # Dictionary to store detected objects for each camera
        self.lock = threading.Lock()
        self.logger = logger

    def update(self, camera_id, x1, y1, x2, y2):
        """
        Updates the common coordinate system with detected object coordinates.

        Args:
            camera_id (int): ID of the camera.
            x1 (float): x-coordinate of the top-left corner of the bounding box.
            y1 (float): y-coordinate of the top-left corner of the bounding box.
            x2 (float): x-coordinate of the bottom-right corner of the bounding box.
            y2 (float): y-coordinate of the bottom-right corner of the bounding box.
        """
        with self.lock:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if camera_id not in self.objects:
                self.objects[camera_id] = {}
                # Initialize object_counter for the camera
                self.objects[camera_id]['object_counter'] = 0

            # Find the closest existing object or create a new one
            closest_object_id, min_distance = None, float('inf')
            for key, value in self.objects[camera_id].items():
                if key == 'object_counter':
                    continue
                if isinstance(value, tuple):
                    object_id, x, y = value
                    distance = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                    if distance < config.MAX_DISTANCE_THRESHOLD and distance < min_distance:
                        closest_object_id, min_distance = object_id, distance

            if closest_object_id is None:
                # Increment object_counter for the camera and assign a new ID
                self.objects[camera_id]['object_counter'] += 1
                closest_object_id = str(self.objects[camera_id]['object_counter'])

            # Update object coordinates and timestamp
            self.objects[camera_id][closest_object_id] = (center_x, center_y, pygame.time.get_ticks())

    def remove_untracked_objects(self, camera_id, tracked_object_ids):
        """
        Removes untracked objects from the common coordinate system.

        Args:
            camera_id (int): ID of the camera.
            tracked_object_ids (set): Set of object IDs that are currently being tracked.
        """
        current_time = pygame.time.get_ticks()
        with self.lock:
            if camera_id in self.objects:
                self.objects[camera_id] = {
                    obj_id: coords for obj_id, coords in self.objects[camera_id].items()
                    if obj_id in tracked_object_ids or current_time - coords[2] < config.OBJECT_TIMEOUT
                }

    def get_triangulated_coordinates(self, use_weighted_average=True):
        with self.lock:
            all_object_coords = []
            for camera_id, objects in self.objects.items():
                for key, value in objects.items():
                    if key == 'object_counter':
                        continue
                    if isinstance(value, tuple):
                        x, y, _ = value
                        all_object_coords.append((key, camera_id, x, y))  # Use key as object_id

            if len(all_object_coords) < 2:
                return []

            object_ids, camera_ids, xs, ys = zip(*all_object_coords)
            unique_object_ids = np.unique(object_ids)

            triangulated_coords = []
            for object_id in unique_object_ids:
                mask = np.array(object_ids) == object_id
                if np.sum(mask) >= 2:
                    try:
                        selected_camera_positions = np.array(config.TRIANGULATION_SETTINGS["CAMERA_POSITIONS"])[
                            np.array(camera_ids)[mask].astype(int)]
                        object_coords = np.array(list(zip(xs, ys)))[mask]
                        triangulated_coord = self.triangulate_multi_camera(
                            object_coords,
                            selected_camera_positions
                        )
                        triangulated_coords.append((object_id, *triangulated_coord))
                    except np.linalg.LinAlgError as e:
                        self.logger.log(f"Error during multi-camera triangulation: {e}")

        if use_weighted_average:
            camera_positions = np.array(config.TRIANGULATION_SETTINGS["CAMERA_POSITIONS"])
            weights = self.calculate_weights(triangulated_coords, camera_positions)
            return self.calculate_weighted_average(triangulated_coords, weights) if triangulated_coords else []
        else:
            return self.calculate_average(triangulated_coords) if triangulated_coords else []

    @staticmethod
    def triangulate_multi_camera(object_coords, camera_positions):
        """Calculates the real-world coordinates using multi-camera triangulation."""
        object_coords = np.array(object_coords)
        a = np.zeros((len(object_coords) * 2, 4))
        b = np.zeros((len(object_coords) * 2, 1))

        for i, (x, y) in enumerate(object_coords):
            a[2 * i] = [x, -1, 0, 0]
            a[2 * i + 1] = [y, 0, -1, 0]
            b[2 * i] = -camera_positions[i, 0]
            b[2 * i + 1] = -camera_positions[i, 1]

        # Solve using least squares method
        solution, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        x, y, z = solution[:3, 0]
        return x, y, z

    @staticmethod
    def calculate_average(coords):
        """Calculates the average of the given coordinates."""
        coords_array = np.array(coords)
        average_coords = np.mean(coords_array, axis=0)
        return list(map(tuple, average_coords))

    @staticmethod
    def calculate_weighted_average(coords, weights):
        """Calculates the weighted average of the given coordinates."""
        object_ids = [coord[0] for coord in coords]
        coords_array = np.array(coords)[:, 1:].astype(np.float64)
        weights_array = np.array(weights)
        weighted_sum = np.sum(coords_array * weights_array[:, np.newaxis], axis=0)
        total_weight = np.sum(weights_array)
        weighted_average_coords = weighted_sum / total_weight
        return [(object_ids[i], *coord) for i, coord in enumerate(weighted_average_coords)]

    @staticmethod
    def calculate_weights(coords, camera_positions):
        """Calculates the weights for each coordinate based on the distance to the cameras."""
        weights = []
        for coord in coords:
            _, x, y, z = coord
            coord_np = np.array([x, y, z])
            distances = np.linalg.norm(camera_positions - coord_np, axis=1)
            weights.append([1 / d for d in distances])
        return weights


class GUI:
    """
    Class to manage the graphical user interface for displaying tracked objects.
    """

    def __init__(self, common_coord_system, logger):
        """
        Initializes the GUI.

        Args:
            common_coord_system (CommonCoordinateSystem): The common coordinate system object.
            logger (AsyncLogger): The logger object for logging messages.
        """
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.GUI_SETTINGS["WINDOW_WIDTH"], config.GUI_SETTINGS["WINDOW_HEIGHT"])
        )
        pygame.display.set_caption(config.GUI_SETTINGS["WINDOW_TITLE"])
        self.clock = pygame.time.Clock()
        self.common_coord_system = common_coord_system
        self.logger = logger
        self.font = pygame.font.Font(None, config.GUI_SETTINGS["FONT_SIZE"])

    def update_gui(self):
        """Updates the GUI with the latest tracked object coordinates and information."""
        self.screen.fill(config.GUI_SETTINGS["BACKGROUND_COLOR"])

        triangulated_coords = self.common_coord_system.get_triangulated_coordinates(use_weighted_average=True)

        for object_id, x, y, z in triangulated_coords:
            # Check if z-coordinate is not zero to avoid division by zero
            if z != 0:
                # Convert 3D coordinates to 2D screen coordinates
                screen_x = int(x / z * config.TRIANGULATION_SETTINGS["SCALE_FACTOR"]) + config.GUI_SETTINGS[
                    "WINDOW_WIDTH"] // 2
                screen_y = int(y / z * config.TRIANGULATION_SETTINGS["SCALE_FACTOR"]) + config.GUI_SETTINGS[
                    "WINDOW_HEIGHT"] // 2

                # Draw a circle to represent the object
                pygame.draw.circle(
                    self.screen, config.GUI_SETTINGS["CIRCLE_COLOR"], (screen_x, screen_y),
                    config.GUI_SETTINGS["CIRCLE_RADIUS"]
                )

                coord_text = self.font.render(
                    f"ID:{object_id} ({x:.2f}, {y:.2f}, {z:.2f})", True, config.GUI_SETTINGS["TEXT_COLOR"]
                )
                self.screen.blit(coord_text, (screen_x + 5, screen_y + 5))

        pygame.display.flip()

    def run(self):
        """
        Runs the GUI event loop.
        """
        self.logger.log("Starting GUI")
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update and render the GUI
            self.update_gui()
            self.clock.tick(config.GUI_SETTINGS["FRAME_RATE"])

        self.logger.log("Stopping GUI")
        pygame.quit()


def main():
    """
    Main function to run the multi-camera tracking system.
    """
    with AsyncLogger() as logger:
        logger.log('Starting main function')

        # Initialize common coordinate system and GUI
        common_coord_system = CommonCoordinateSystem(logger)
        gui = GUI(common_coord_system, logger)

        # Create queues for each video file and start video processors
        queues = [Queue() for _ in config.VIDEO_FILES]
        video_processors = [
            VideoProcessor(video_path, queues[i], i, logger)
            for i, video_path in enumerate(config.VIDEO_FILES)
        ]
        for vp in video_processors:
            vp.start()

        # Create events for stopping and syncing threads
        stop_event = threading.Event()
        queue_event = threading.Event()

        def process_queue():
            """
            Processes the queue of detected object coordinates.
            """
            logger.log('Starting queue processing')
            while not stop_event.is_set():
                for q in queues:
                    while not q.empty():
                        camera_id, x1, y1, x2, y2 = q.get()
                        common_coord_system.update(camera_id, x1, y1, x2, y2)
                queue_event.wait(timeout=config.QUEUE_PROCESS_DELAY)
                queue_event.clear()
            logger.log('Stopping queue processing')

        # Start queue processing thread
        queue_thread = threading.Thread(target=process_queue)
        queue_thread.daemon = True
        queue_thread.start()

        # Run the GUI
        gui.run()

        # Stop all threads
        stop_event.set()
        queue_event.set()
        for vp in video_processors:
            vp.stop()
            vp.join()
        queue_thread.join()

        logger.log('Stopping main function')


if __name__ == "__main__":
    main()
