import threading
import time
from queue import Queue

import cv2
import numpy as np
import pygame
import torch
import torchreid
from filterpy.kalman import KalmanFilter
from sklearn.metrics.pairwise import cosine_similarity
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
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'Unable to open video file: {video_path}')
        self.model = YOLO(config.YOLO_MODEL_PATH)

    def run(self):
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
        self.cap.release()

    def process_frame(self, frame):
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            person_img = frame[int(y1):int(y2), int(x1):int(x2)]
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            person_img_tensor = torch.from_numpy(person_img).float().permute(2, 0, 1).unsqueeze(0).cuda()
            with torch.no_grad():
                feature = self.reid_model(person_img_tensor)
            self.queue.put((self.camera_id, x1, y1, x2, y2, feature.cpu().numpy()))

    def stop(self):
        self.stop_event.set()


class CommonCoordinateSystem:
    def __init__(self, logger):
        self.objects = {}
        self.next_id = 0
        self.lock = threading.Lock()
        self.logger = logger

    def update(self, camera_id, x1, y1, x2, y2, feature):
        with self.lock:
            matched = self.match_or_create_object(camera_id, x1, y1, x2, y2, feature)
            if not matched:
                self.create_new_object(camera_id, x1, y1, x2, y2, feature)

    def match_or_create_object(self, camera_id, x1, y1, x2, y2, feature):
        min_cost = float('inf')
        best_match_id = None

        for obj_id, data in self.objects.items():
            cost = self.calculate_cost(data['feature'], feature, data['boxes'][-1], (x1, y1, x2, y2))
            if cost < min_cost and cost < config.COST_THRESHOLD:
                min_cost = cost
                best_match_id = obj_id

        if best_match_id is not None:
            self.objects[best_match_id]['boxes'].append((camera_id, x1, y1, x2, y2))
            self.objects[best_match_id]['feature'] = self.update_feature(self.objects[best_match_id]['feature'],
                                                                         feature)
            self.objects[best_match_id]['kf'].update(np.array([x1, y1, x2, y2]))
            return True

        return False

    @staticmethod
    def calculate_cost(feature1, feature2, box1, box2):
        similarity = cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))[0][0]
        feature_cost = 1 - similarity

        center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        distance_cost = np.linalg.norm(center1 - center2)

        return feature_cost + distance_cost

    @staticmethod
    def update_feature(old_feature, new_feature, alpha=0.5):
        return alpha * new_feature + (1 - alpha) * old_feature

    def create_new_object(self, camera_id, x1, y1, x2, y2, feature):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        kf.R *= 10.
        kf.P *= 10.
        kf.Q *= 0.01
        kf.x[:4] = np.array([x1, y1, x2, y2]).reshape((4, 1))

        self.objects[self.next_id] = {
            'boxes': [(camera_id, x1, y1, x2, y2)],
            'feature': feature,
            'kf': kf
        }
        self.next_id += 1

    def get_objects(self):
        with self.lock:
            return self.objects.copy()


class GUI:
    def __init__(self, common_coord_system, logger):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.GUI_SETTINGS['WINDOW_WIDTH'], config.GUI_SETTINGS['WINDOW_HEIGHT']))
        pygame.display.set_caption(config.GUI_SETTINGS['WINDOW_TITLE'])
        self.clock = pygame.time.Clock()
        self.common_coord_system = common_coord_system
        self.logger = logger
        self.colors = {}

    def get_color(self, obj_id):
        if obj_id not in self.colors:
            self.colors[obj_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        return self.colors[obj_id]

    def update_gui(self):
        self.screen.fill(config.GUI_SETTINGS['BACKGROUND_COLOR'])
        objects = self.common_coord_system.get_objects()
        for obj_id, data in objects.items():
            color = self.get_color(obj_id)
            last_center = None
            for (camera_id, x1, y1, x2, y2) in data['boxes']:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)),
                                   config.GUI_SETTINGS['CIRCLE_RADIUS'])
                if last_center is not None:
                    pygame.draw.line(self.screen, color, last_center, (int(center_x), int(center_y)), 2)
                last_center = (int(center_x), int(center_y))
                font = pygame.font.SysFont('Arial', 12)
                text_surface = font.render(f'ID: {obj_id}', True, color)
                self.screen.blit(text_surface, (int(center_x) + 5, int(center_y) - 10))
        pygame.display.flip()

    def run(self):
        self.logger.log('Starting GUI')
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.update_gui()
            self.clock.tick(config.GUI_SETTINGS['FRAME_RATE'])
        self.logger.log('Stopping GUI')
        pygame.quit()


def process_queue(queues, common_coord_system, stop_event, logger):
    logger.log('Starting queue processing')
    while not stop_event.is_set():
        for q in queues:
            while not q.empty():
                camera_id, x1, y1, x2, y2, feature = q.get()
                common_coord_system.update(camera_id, x1, y1, x2, y2, feature)
        time.sleep(config.QUEUE_PROCESS_DELAY)
    logger.log('Stopping queue processing')


def main():
    with AsyncLogger() as logger:
        logger.log('Starting main function')

        common_coord_system = CommonCoordinateSystem(logger)
        gui = GUI(common_coord_system, logger)

        reid_model = torchreid.models.build_model(
            name='resnet50',
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
