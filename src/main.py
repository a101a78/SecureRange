import threading
import time
from queue import Queue

import cv2
import numpy as np
import pygame
from ultralytics import YOLO

from src import config


class VideoProcessor(threading.Thread):
    def __init__(self, video_path, queue, camera_id):
        super().__init__()
        self.video_path = video_path
        self.queue = queue
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.camera_id = camera_id
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set() and self.cap.isOpened():
            for _ in range(config.FRAME_SKIP):
                self.cap.read()
            success, frame = self.cap.read()
            if not success:
                break
            results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD)
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                self.queue.put((self.camera_id, x1, y1, x2, y2, time.time()))
        self.cap.release()

    def stop(self):
        self.stop_event.set()


class CommonCoordinateSystem:
    def __init__(self):
        self.objects = {}
        self.next_id = 0
        self.lock = threading.Lock()

    def update(self, camera_id, x1, y1, x2, y2, timestamp):
        with self.lock:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            common_x = center_x / config.COMMON_COORDINATE_SYSTEM_SCALE
            common_y = center_y / config.COMMON_COORDINATE_SYSTEM_SCALE
            matched = False

            for obj_id, data in self.objects.items():
                dist = np.sqrt((data['cx'] - common_x) ** 2 + (data['cy'] - common_y) ** 2)
                if dist < config.COORDINATE_MATCH_THRESHOLD:
                    self.objects[obj_id]['boxes'].append((camera_id, x1, y1, x2, y2, timestamp))
                    self.objects[obj_id]['cx'] = common_x
                    self.objects[obj_id]['cy'] = common_y
                    matched = True
                    break

            if not matched:
                self.objects[self.next_id] = {
                    'boxes': [(camera_id, x1, y1, x2, y2, timestamp)],
                    'cx': common_x,
                    'cy': common_y
                }
                self.next_id += 1

    def get_objects(self):
        with self.lock:
            return self.objects


class GUI:
    def __init__(self, common_coord_system):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.GUI_SETTINGS["WINDOW_WIDTH"], config.GUI_SETTINGS["WINDOW_HEIGHT"]))
        pygame.display.set_caption(config.GUI_SETTINGS["WINDOW_TITLE"])
        self.clock = pygame.time.Clock()
        self.common_coord_system = common_coord_system
        self.hovered_id = None

    def update_gui(self):
        self.screen.fill(config.GUI_SETTINGS["BACKGROUND_COLOR"])
        objects = self.common_coord_system.get_objects()
        for obj_id, data in objects.items():
            for (camera_id, x1, y1, x2, y2, timestamp) in data['boxes']:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                pygame.draw.circle(self.screen, config.GUI_SETTINGS["CIRCLE_COLOR"], (int(center_x), int(center_y)),
                                   config.GUI_SETTINGS["CIRCLE_RADIUS"])
                if self.hovered_id == obj_id:
                    font = pygame.font.Font(None, config.GUI_SETTINGS["FONT_SIZE"])
                    text = font.render(f"ID: {obj_id}", True, config.GUI_SETTINGS["TEXT_COLOR"])
                    self.screen.blit(text, (int(center_x), int(center_y) - 10))
        pygame.display.flip()

    def check_hover(self, pos):
        self.hovered_id = None
        objects = self.common_coord_system.get_objects()
        for obj_id, data in objects.items():
            for camera_id, x1, y1, x2, y2, timestamp in data['boxes']:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if center_x - 3 < pos[0] < center_x + 3 and center_y - 3 < pos[1] < center_y + 3:
                    self.hovered_id = obj_id
                    break
            if self.hovered_id is not None:
                break

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    self.check_hover(event.pos)
            self.update_gui()
            self.clock.tick(config.GUI_SETTINGS["FRAME_RATE"])
        pygame.quit()


def main():
    common_coord_system = CommonCoordinateSystem()
    gui = GUI(common_coord_system)

    queues = [Queue() for _ in config.VIDEO_FILES]
    video_processors = [VideoProcessor(video_path, queues[i], i) for i, video_path in enumerate(config.VIDEO_FILES)]

    for vp in video_processors:
        vp.start()

    stop_event = threading.Event()

    def process_queue():
        while not stop_event.is_set():
            for q in queues:
                while not q.empty():
                    camera_id, x1, y1, x2, y2, timestamp = q.get()
                    common_coord_system.update(camera_id, x1, y1, x2, y2, timestamp)
            time.sleep(config.QUEUE_PROCESS_DELAY)

    queue_thread = threading.Thread(target=process_queue)
    queue_thread.daemon = True
    queue_thread.start()

    gui.run()

    stop_event.set()
    for vp in video_processors:
        vp.stop()
        vp.join()

    queue_thread.join()


if __name__ == "__main__":
    main()
