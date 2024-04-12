import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

import config

# YOLOv8 모델 로드
model = YOLO(config.YOLO_MODEL_PATH)


def stream_receiver(url, frames_queue, stop_event):
    """
    RTSP 스트림에서 비디오 프레임을 수신하여 frames_queue에 추가하는 함수.

    Args:
        url (str): RTSP 스트림 URL.
        frames_queue (list): 수신된 비디오 프레임을 저장할 리스트.
        stop_event (threading.Event): 스레드 종료를 알리는 이벤트.
    """
    cap = cv2.VideoCapture(url)

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            retry_count = 0
            while not ret and retry_count < config.MAX_RETRIES:
                print(
                    f"{url}에서 프레임 읽기 실패. {config.RETRY_DELAY}초 후 재시도... (시도 {retry_count + 1}/{config.MAX_RETRIES})")
                time.sleep(config.RETRY_DELAY)
                ret, frame = cap.read()
                retry_count += 1

            if not ret:
                print(f"{url}에서 {config.MAX_RETRIES}번 시도 후에도 프레임 읽기 실패. 스트림을 중지합니다.")
                break

        frames_queue.append((url, frame))

    cap.release()


def main():
    """
    비디오 스트림을 처리하고, 객체 추적을 수행하며, 여러 카메라에서 동일 인물을 식별하는 메인 함수.
    """
    frames_queue = []
    stop_event = threading.Event()

    # 각 RTSP 스트림에 대한 스레드 생성 및 시작
    threads = []
    for i, url in enumerate(config.RTSP_URLS):
        t = threading.Thread(target=stream_receiver, args=(url, frames_queue, stop_event))
        threads.append(t)
        t.start()

    prev_matched_indices = [[] for _ in range(len(config.RTSP_URLS))]  # 이전 프레임의 매칭 결과 저장

    while True:
        if len(frames_queue) >= len(config.RTSP_URLS):
            # frames_queue에서 프레임 추출
            frames = [frame for _, frame in frames_queue[:len(config.RTSP_URLS)]]

            # YOLOv8과 BoT-SORT를 사용하여 각 프레임에 대해 객체 추적 수행
            track_results = [model.track(frame, persist=True)[0] for frame in frames]

            # 추적된 결과에서 각 사람에 대한 외관 특징 추출
            features_list = []
            for result in track_results:
                features = [r.appearance for r in result if r.cls == 0]
                features_list.append(features)

            # 외관 특징을 기반으로 프레임 간 사람 매칭
            matched_indices = [[] for _ in range(len(config.RTSP_URLS))]  # 현재 프레임의 매칭 결과 저장
            for i in range(1, len(features_list)):
                if len(features_list[0]) > 0 and len(features_list[i]) > 0:
                    cos_sim = np.dot(features_list[0], np.array(features_list[i]).T)
                    indices = np.argmax(cos_sim, axis=1)
                    matched_indices[i] = indices.tolist()
                else:
                    matched_indices[i] = []

            # 매칭 결과를 프레임에 시각화
            for i, result in enumerate(track_results):
                frame = result.orig_img
                for j, match_idx in enumerate(matched_indices[i]):
                    if match_idx >= 0 and match_idx < len(features_list[0]):
                        person1 = track_results[0].boxes.xyxy[j].tolist()
                        person2 = result.boxes.xyxy[match_idx].tolist()

                        x1, y1, x2, y2 = map(int, person1)
                        cv2.rectangle(track_results[0].orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        x1, y1, x2, y2 = map(int, person2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 이전 프레임에서 매칭되었던 인덱스와 현재 프레임의 매칭 인덱스가 다른 경우, 선 그리기
                        if i > 0 and j < len(prev_matched_indices[i]) and prev_matched_indices[i][j] != match_idx:
                            prev_person = track_results[i - 1].boxes.xyxy[prev_matched_indices[i][j]].tolist()
                            curr_person = person2

                            prev_x, prev_y = (prev_person[0] + prev_person[2]) / 2, (
                                    prev_person[1] + prev_person[3]) / 2
                            curr_x, curr_y = (curr_person[0] + curr_person[2]) / 2, (
                                    curr_person[1] + curr_person[3]) / 2

                            cv2.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), (0, 0, 255), 2)

                cv2.imshow(f'RTSP Stream {i + 1}', frame)

            prev_matched_indices = matched_indices  # 현재 프레임의 매칭 결과를 이전 프레임 매칭 결과로 업데이트

            # 처리된 프레임을 frames_queue에서 제거
            frames_queue = frames_queue[len(config.RTSP_URLS):]

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
