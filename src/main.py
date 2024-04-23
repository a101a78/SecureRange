import threading

import cv2
import numpy as np
from ultralytics import YOLO

import config

# YOLOv8 모델 로드
model = YOLO(config.YOLO_MODEL_PATH)


def video_reader(file_path, frames_dict, stop_event):
    """
    비디오 파일에서 프레임을 읽어 frames_dict에 추가합니다.

    Args:
        file_path (str): 비디오 파일의 경로.
        frames_dict (dict): 읽은 비디오 프레임을 저장할 딕셔너리.
        stop_event (threading.Event): 스레드 종료 이벤트.
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print(f'Error opening video file: {file_path}')
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        frames_dict[file_path].append(frame)

    cap.release()


def calculate_similarity(feat1, feat2, size1, size2):
    """
    두 특징 벡터 간의 유사도를 계산합니다.

    Args:
        feat1: 첫 번째 특징 벡터.
        feat2: 두 번째 특징 벡터.
        size1: 첫 번째 사람의 크기 (면적).
        size2: 두 번째 사람의 크기 (면적).

    Returns:
        float: 유사도 점수.
    """
    # 코사인 유사도 계산
    cos_sim = np.dot(feat1, feat2.T) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    cos_sim = cos_sim.item()  # 스칼라 값으로 변환

    # 크기 유사도 계산
    size_sim = 1 - np.abs(size1 - size2) / np.maximum(size1, size2)

    # 유사도 점수 조합
    similarity = 0.7 * cos_sim + 0.3 * size_sim

    return similarity


def draw_rectangle(frame, person, color, object_id):
    """
    프레임에 사각형과 객체 번호를 그립니다.

    Args:
        frame: 사각형을 그릴 프레임.
        person: 사각형을 그릴 사람의 좌표 정보.
        color: 사각형의 색상.
        object_id: 객체의 번호.
    """
    x1, y1, x2, y2 = map(int, person)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, str(object_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def get_scale(frame_width, frame_height, orig_shape):
    """
    프레임 크기와 원본 이미지 크기를 기반으로 스케일 비율을 계산합니다.

    Args:
        frame_width (int): 프레임의 너비.
        frame_height (int): 프레임의 높이.
        orig_shape (tuple): 원본 이미지의 크기 (높이, 너비).

    Returns:
        tuple: 너비와 높이에 대한 스케일 비율 (scale_x, scale_y).
    """
    orig_height, orig_width = orig_shape[:2]
    scale_x = frame_width / orig_width
    scale_y = frame_height / orig_height
    return scale_x, scale_y


def extract_features(track_results):
    """
    탐지된 결과에서 각 사람에 대한 외관 특징을 추출합니다.

    Args:
        track_results (list): YOLOv8 모델의 탐지 결과.

    Returns:
        tuple: (features_list, sizes_list, max_feature_length)
            - features_list (list): 각 프레임별 사람들의 외관 특징 리스트.
            - sizes_list (list): 각 프레임별 사람들의 크기 리스트.
            - max_feature_length (int): 가장 긴 특징 벡터의 길이.
    """
    features_list = []
    sizes_list = []
    max_feature_length = 0

    for result in track_results:
        features = []
        sizes = []
        for r in result:
            persons = [detection for detection in r.boxes.data.tolist() if detection[-1] == 0]
            person_features = []
            person_sizes = []
            for person in persons:
                x1, y1, x2, y2 = map(int, person[:4])
                person_img = r.orig_img[y1:y2, x1:x2]
                person_result = model(person_img)
                feature = person_result[0].boxes.conf.tolist()
                person_features.extend(feature)
                person_sizes.append((x2 - x1) * (y2 - y1))
            features.append(person_features)
            sizes.append(max(person_sizes) if person_sizes else 0)
            max_feature_length = max(max_feature_length, len(person_features))

        features_list.append(features)
        sizes_list.append(sizes)

    return features_list, sizes_list, max_feature_length


def match_persons(padded_features_list, sizes_list):
    """
    외관 특징을 기반으로 프레임 간 사람을 매칭합니다.

    Args:
        padded_features_list (list): 패딩된 외관 특징 리스트.
        sizes_list (list): 각 프레임별 사람들의 크기 리스트.

    Returns:
        list: 매칭된 인덱스 정보.
    """
    num_frames = len(padded_features_list)
    matched_indices = [[[] for _ in range(num_frames)] for _ in range(num_frames)]

    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            if len(padded_features_list[i]) > 0 and len(padded_features_list[j]) > 0:
                cos_sim = np.zeros((len(padded_features_list[i]), len(padded_features_list[j])))
                for k in range(len(padded_features_list[i])):
                    feat1 = np.array(padded_features_list[i][k]).reshape(1, -1)
                    for m in range(len(padded_features_list[j])):
                        if matched_indices[j][i]:
                            continue
                        feat2 = np.array(padded_features_list[j][m]).reshape(1, -1)
                        size1 = sizes_list[i][k]
                        size2 = sizes_list[j][m]
                        cos_sim[k, m] = calculate_similarity(feat1, feat2, size1, size2)
                indices = np.argmax(cos_sim, axis=1)
                matched_indices[i][j] = indices.tolist()
                matched_indices[j][i] = [np.argmax(cos_sim, axis=0)[x] for x in indices]
            else:
                matched_indices[i][j] = []
                matched_indices[j][i] = []

    return matched_indices


def visualize_results(track_results, matched_indices):
    """
    매칭 결과를 프레임에 시각화합니다.

    Args:
        track_results (list): YOLOv8 모델의 탐지 결과.
        matched_indices (list): 매칭된 인덱스 정보.
    """
    frame_width = config.FRAME_SIZE['w']
    frame_height = config.FRAME_SIZE['h']

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i in range(len(track_results)):
        frame = cv2.resize(track_results[i][0].orig_img, (frame_width, frame_height))
        persons = track_results[i][0].boxes.data.tolist()

        for j, person in enumerate(persons):
            scale_x, scale_y = get_scale(frame_width, frame_height, track_results[i][0].orig_img.shape)
            person_coords = [person[0] * scale_x, person[1] * scale_y, person[2] * scale_x, person[3] * scale_y]
            color = colors[j % len(colors)]
            draw_rectangle(frame, person_coords, color, j + 1)

            for k in range(i + 1, len(track_results)):
                if j < len(matched_indices[i][k]):
                    match_idx = matched_indices[i][k][j]
                    if match_idx < len(persons):
                        match_person = persons[match_idx]
                        scale_x_match, scale_y_match = get_scale(frame_width, frame_height,
                                                                 track_results[k][0].orig_img.shape)
                        match_coords = [match_person[0] * scale_x_match, match_person[1] * scale_y_match,
                                        match_person[2] * scale_x_match, match_person[3] * scale_y_match]
                        curr_x, curr_y = (person_coords[0] + person_coords[2]) / 2, (
                                person_coords[1] + person_coords[3]) / 2
                        match_x, match_y = (match_coords[0] + match_coords[2]) / 2, (
                                match_coords[1] + match_coords[3]) / 2
                        cv2.line(frame, (int(curr_x), int(curr_y)), (int(match_x), int(match_y)), color, 2)

        cv2.imshow(f'Video {i + 1}', frame)


def print_matching_info(matched_indices):
    """
    매칭 정보를 출력합니다.

    Args:
        matched_indices (list): 매칭된 인덱스 정보.
    """
    num_frames = len(matched_indices)

    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            for k, match_idx in enumerate(matched_indices[i][j]):
                print(f'프레임 {i + 1}의 {k + 1}번째 사람은 프레임 {j + 1}의 {match_idx + 1}번째 사람과 동일')

    print('---')


def main():
    frames_dict = {file_path: [] for file_path in config.VIDEO_FILES}
    stop_event = threading.Event()

    # 각 비디오 파일에 대한 스레드 생성 및 시작
    threads = []
    for i, file_path in enumerate(config.VIDEO_FILES):
        t = threading.Thread(target=video_reader, args=(file_path, frames_dict, stop_event))
        threads.append(t)
        t.start()

    while True:
        if all(len(frames) > 0 for frames in frames_dict.values()):
            # frames_dict에서 프레임 추출
            frames = [frames_dict[file_path].pop(0) for file_path in config.VIDEO_FILES]

            # YOLOv8을 사용하여 각 프레임에 대해 객체 탐지 수행
            track_results = [model(frame, device=0, classes=0, conf=0.6) for frame in frames]

            # 탐지된 결과에서 각 사람에 대한 외관 특징 추출
            features_list, sizes_list, max_feature_length = extract_features(track_results)

            # 특징 벡터의 차원을 통일
            padded_features_list = []
            for features in features_list:
                padded_features = []
                for feature in features:
                    padded_feature = feature + [0] * (max_feature_length - len(feature))
                    padded_features.append(padded_feature)
                padded_features_list.append(padded_features)

            # 외관 특징을 기반으로 프레임 간 사람 매칭
            matched_indices = match_persons(padded_features_list, sizes_list)

            # 매칭 정보 출력
            # print_matching_info(matched_indices)

            # 매칭 결과를 프레임에 시각화
            visualize_results(track_results, matched_indices)

        # 'q' 키 입력 시 프로그램 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # 모든 스레드 종료 대기
    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
