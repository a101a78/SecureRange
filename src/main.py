import threading

import cv2
import numpy as np
from ultralytics import YOLO

import config

# YOLOv8 모델 로드
model = YOLO(config.YOLO_MODEL_PATH)


def video_reader(file_path, frames_queue, stop_event):
    """
    비디오 파일에서 프레임을 읽어 frames_queue에 추가하는 함수.

    Args:
        file_path (str): 비디오 파일 경로.
        frames_queue (list): 읽은 비디오 프레임을 저장할 리스트.
        stop_event (threading.Event): 스레드 종료를 알리는 이벤트.
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print(f"Error opening video file: {file_path}")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        frames_queue.append((file_path, frame))

    cap.release()


def main():
    """
    비디오 파일을 처리하고, 객체 추적을 수행하며, 여러 비디오에서 동일 인물을 식별하는 메인 함수.
    """
    frames_queue = []
    stop_event = threading.Event()

    # 각 비디오 파일에 대한 스레드 생성 및 시작
    threads = []
    for i, file_path in enumerate(config.VIDEO_FILES):
        t = threading.Thread(target=video_reader, args=(file_path, frames_queue, stop_event))
        threads.append(t)
        t.start()

    prev_matched_indices = [[] for _ in range(len(config.VIDEO_FILES))]  # 이전 프레임의 매칭 결과 저장

    while True:
        if len(frames_queue) >= len(config.VIDEO_FILES):
            # frames_queue에서 프레임 추출
            frames = [frame for _, frame in frames_queue[:len(config.VIDEO_FILES)]]

            # YOLOv8을 사용하여 각 프레임에 대해 객체 탐지 수행
            track_results = [model(frame) for frame in frames]

            # 탐지된 결과에서 각 사람에 대한 외관 특징 추출
            features_list = []
            max_feature_length = 0
            for result in track_results:
                features = []
                for r in result:
                    persons = [detection for detection in r.boxes.data.tolist() if detection[-1] == 0]
                    person_features = []
                    for person in persons:
                        x1, y1, x2, y2 = map(int, person[:4])
                        person_img = r.orig_img[y1:y2, x1:x2]
                        person_result = model(person_img)
                        feature = person_result[0].boxes.conf.tolist()  # 예측 결과의 신뢰도를 특징으로 사용
                        person_features.extend(feature)  # 특징 벡터를 1차원으로 펼침
                    features.append(person_features)
                    max_feature_length = max(max_feature_length, len(person_features))
                features_list.append(features)

            # 특징 벡터의 차원을 통일
            padded_features_list = []
            for features in features_list:
                padded_features = []
                for feature in features:
                    padded_feature = feature + [0] * (max_feature_length - len(feature))
                    padded_features.append(padded_feature)
                padded_features_list.append(padded_features)

            # 외관 특징을 기반으로 프레임 간 사람 매칭
            matched_indices = [[] for _ in range(len(config.VIDEO_FILES))]  # 현재 프레임의 매칭 결과 저장
            for i in range(1, len(padded_features_list)):
                if len(padded_features_list[0]) > 0 and len(padded_features_list[i]) > 0:
                    cos_sim = np.zeros((len(padded_features_list[0]), len(padded_features_list[i])))
                    for j in range(len(padded_features_list[0])):
                        for k in range(len(padded_features_list[i])):
                            feat1 = np.array(padded_features_list[0][j]).reshape(1, -1)
                            feat2 = np.array(padded_features_list[i][k]).reshape(1, -1)
                            cos_sim[j, k] = np.dot(feat1, feat2.T) / (
                                    np.linalg.norm(feat1, axis=1) * np.linalg.norm(feat2, axis=1))
                    indices = np.argmax(cos_sim, axis=1)
                    matched_indices[i] = indices.tolist()
                else:
                    matched_indices[i] = []

            # 매칭 결과를 프레임에 시각화
            for i, result in enumerate(track_results):
                frame = result[0].orig_img
                for j, match_idx in enumerate(matched_indices[i]):
                    if 0 <= match_idx < len(features_list[0]):
                        person1 = track_results[0][0].boxes.xyxy[j].tolist()
                        person2 = result[0].boxes.xyxy[match_idx].tolist()

                        x1, y1, x2, y2 = map(int, person1)
                        cv2.rectangle(track_results[0][0].orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        x1, y1, x2, y2 = map(int, person2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 이전 프레임에서 매칭되었던 인덱스와 현재 프레임의 매칭 인덱스가 다른 경우, 선 그리기
                        if i > 0 and j < len(prev_matched_indices[i]) and prev_matched_indices[i][j] != match_idx:
                            prev_person = track_results[i - 1][0].boxes.xyxy[prev_matched_indices[i][j]].tolist()
                            curr_person = person2

                            prev_x, prev_y = (prev_person[0] + prev_person[2]) / 2, (
                                    prev_person[1] + prev_person[3]) / 2
                            curr_x, curr_y = (curr_person[0] + curr_person[2]) / 2, (
                                    curr_person[1] + curr_person[3]) / 2

                            cv2.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), (0, 0, 255), 2)

                cv2.imshow(f'Video {i + 1}', frame)

            prev_matched_indices = matched_indices  # 현재 프레임의 매칭 결과를 이전 프레임 매칭 결과로 업데이트

            # 처리된 프레임을 frames_queue에서 제거
            frames_queue = frames_queue[len(config.VIDEO_FILES):]

        # 'q' 키 입력 시 프로그램 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # 모든 스레드 종료 대기
    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
