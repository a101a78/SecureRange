import threading
import time
import tkinter as tk
from queue import Queue

import cv2
import numpy as np
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
    def __init__(self, root, common_coord_system):
        self.root = root
        self.common_coord_system = common_coord_system
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        self.update_gui()
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.hovered_id = None

    def update_gui(self):
        self.canvas.delete("all")
        objects = self.common_coord_system.get_objects()
        current_time = time.time()
        for obj_id, data in objects.items():
            valid_boxes = [box for box in data['boxes'] if current_time - box[5] <= config.TRAJECTORY_DWELL_TIME]
            data['boxes'] = valid_boxes
            for i, (camera_id, x1, y1, x2, y2, timestamp) in enumerate(valid_boxes):
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                self.canvas.create_oval(center_x - 3, center_y - 3, center_x + 3, center_y + 3, fill="red")
                if i > 0:
                    prev_center_x = (valid_boxes[i - 1][1] + valid_boxes[i - 1][3]) / 2
                    prev_center_y = (valid_boxes[i - 1][2] + valid_boxes[i - 1][4]) / 2
                    self.canvas.create_line(prev_center_x, prev_center_y, center_x, center_y, fill="blue")
                if self.hovered_id == obj_id:
                    self.canvas.create_text(center_x, center_y - 10, text=f"ID: {obj_id}", fill="blue")
        self.root.after(100, self.update_gui)

    def on_mouse_move(self, event):
        self.hovered_id = None
        objects = self.common_coord_system.get_objects()
        for obj_id, data in objects.items():
            for camera_id, x1, y1, x2, y2, timestamp in data['boxes']:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if center_x - 3 < event.x < center_x + 3 and center_y - 3 < event.y < center_y + 3:
                    self.hovered_id = obj_id
                    break
            if self.hovered_id is not None:
                break


def main():
    root = tk.Tk()
    root.title("Multi-Camera Tracking System")
    common_coord_system = CommonCoordinateSystem()
    GUI(root, common_coord_system)

    queues = [Queue() for _ in config.VIDEO_FILES]
    video_processors = [VideoProcessor(video_path, queues[i], i) for i, video_path in enumerate(config.VIDEO_FILES)]

    for vp in video_processors:
        vp.start()

    def process_queue():
        for q in queues:
            while not q.empty():
                camera_id, x1, y1, x2, y2, timestamp = q.get()
                common_coord_system.update(camera_id, x1, y1, x2, y2, timestamp)
        root.after(50, process_queue)

    process_queue()
    root.mainloop()

    for vp in video_processors:
        vp.stop()
        vp.join()


if __name__ == "__main__":
    main()
