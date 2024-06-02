import threading
import time
from queue import Queue

import cv2
import numpy as np
import pygame
import torch
import torchreid
from ultralytics import YOLO

from src import config
from utils.logger import AsyncLogger


class VideoProcessor(threading.Thread):
    def __init__(self, video_path, queue, camera_id, logger, reid_model):
        super().__init__()
        self.video_path = video_path
        self.queue = queue
        self.camera_id = camera_id
        self.logger = logger
        self.reid_model = reid_model
        self.stop_event = threading.Event()
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f'Unable to open video file: {video_path}')
            self.model = YOLO(config.YOLO_MODEL_PATH)
        except Exception as e:
            self.logger.log(f'Error initializing VideoProcessor for camera {self.camera_id}: {e}')
            raise

    def run(self):
        try:
            self.logger.log(f'Starting video processing for camera {self.camera_id}')
            while not self.stop_event.is_set() and self.cap.isOpened():
                for _ in range(config.FRAME_SKIP):
                    self.cap.read()
                success, frame = self.cap.read()
                if not success:
                    self.logger.log(f'Failed to read frame for camera {self.camera_id}')
                    break
                self.process_frame(frame)
            self.logger.log(f'Stopping video processing for camera {self.camera_id}')
        except Exception as e:
            self.logger.log(f'Error during video processing for camera {self.camera_id}: {e}')
        finally:
            self.cap.release()

    def process_frame(self, frame):
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            person_img = frame[int(y1):int(y2), int(x1):int(x2)]
            person_img_tensor = torch.from_numpy(person_img).float().permute(2, 0, 1).unsqueeze(0).cuda()
            with torch.no_grad():
                feature = self.reid_model(person_img_tensor)
            self.queue.put((self.camera_id, x1, y1, x2, y2, feature.cpu().numpy(), time.time()))

    def stop(self):
        self.stop_event.set()


class CommonCoordinateSystem:
    def __init__(self, logger):
        self.objects = {}
        self.next_id = 0
        self.lock = threading.Lock()
        self.logger = logger

    def update(self, camera_id, x1, y1, x2, y2, feature, timestamp):
        try:
            with self.lock:
                matched = self.match_or_create_object(camera_id, x1, y1, x2, y2, feature, timestamp)
                if not matched:
                    self.create_new_object(camera_id, x1, y1, x2, y2, feature, timestamp)
        except Exception as e:
            self.logger.log(f'Error updating coordinates: {e}')

    def match_or_create_object(self, camera_id, x1, y1, x2, y2, feature, timestamp):
        matched = False
        for obj_id, data in self.objects.items():
            if self.is_same_person(data['feature'], feature):
                self.objects[obj_id]['boxes'].append((camera_id, x1, y1, x2, y2, timestamp))
                self.objects[obj_id]['feature'] = feature
                matched = True
                break
        return matched

    @staticmethod
    def is_same_person(feature1, feature2):
        dist = np.linalg.norm(feature1 - feature2)
        return dist < config.FEATURE_MATCH_THRESHOLD

    def create_new_object(self, camera_id, x1, y1, x2, y2, feature, timestamp):
        self.objects[self.next_id] = {
            'boxes': [(camera_id, x1, y1, x2, y2, timestamp)],
            'feature': feature
        }
        self.next_id += 1

    def get_objects(self):
        with self.lock:
            return self.objects


class GUI:
    def __init__(self, common_coord_system, logger):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.GUI_SETTINGS['WINDOW_WIDTH'], config.GUI_SETTINGS['WINDOW_HEIGHT']))
        pygame.display.set_caption(config.GUI_SETTINGS['WINDOW_TITLE'])
        self.clock = pygame.time.Clock()
        self.common_coord_system = common_coord_system
        self.logger = logger

    def update_gui(self):
        self.screen.fill(config.GUI_SETTINGS['BACKGROUND_COLOR'])
        objects = self.common_coord_system.get_objects()
        for obj_id, data in objects.items():
            for (camera_id, x1, y1, x2, y2, timestamp) in data['boxes']:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                pygame.draw.circle(self.screen, config.GUI_SETTINGS['CIRCLE_COLOR'], (int(center_x), int(center_y)),
                                   config.GUI_SETTINGS['CIRCLE_RADIUS'])
        pygame.display.flip()

    def run(self):
        try:
            self.logger.log('Starting GUI')
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                self.update_gui()
                self.clock.tick(config.GUI_SETTINGS['FRAME_RATE'])
            self.logger.log('Stopping GUI')
        except Exception as e:
            self.logger.log(f'Error in GUI: {e}')
        finally:
            pygame.quit()


def process_queue(queues, common_coord_system, stop_event, logger):
    try:
        logger.log('Starting queue processing')
        while not stop_event.is_set():
            for q in queues:
                while not q.empty():
                    camera_id, x1, y1, x2, y2, feature, timestamp = q.get()
                    common_coord_system.update(camera_id, x1, y1, x2, y2, feature, timestamp)
            time.sleep(config.QUEUE_PROCESS_DELAY)
        logger.log('Stopping queue processing')
    except Exception as e:
        logger.log(f'Error processing queue: {e}')


def main():
    with AsyncLogger() as logger:
        logger.log('Starting main function')

        common_coord_system = CommonCoordinateSystem(logger)
        gui = GUI(common_coord_system, logger)

        reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            loss='softmax',
            pretrained=True
        ).cuda()
        reid_model.eval()

        queues = [Queue() for _ in config.VIDEO_FILES]
        video_processors = [VideoProcessor(video_path, queues[i], i, logger, reid_model) for i, video_path in
                            enumerate(config.VIDEO_FILES)]

        for vp in video_processors:
            vp.start()

        stop_event = threading.Event()
        queue_thread = threading.Thread(target=process_queue,
                                        args=(queues, common_coord_system, stop_event, logger))
        queue_thread.daemon = True
        queue_thread.start()

        gui.run()

        stop_event.set()
        for vp in video_processors:
            vp.stop()
            vp.join()

        queue_thread.join()

        logger.log('Stopping main function')


if __name__ == '__main__':
    main()
