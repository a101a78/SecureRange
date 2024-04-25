import threading

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

import config

model = YOLO(config.YOLO_MODEL_PATH)


def video_reader(file_path, frames_dict, stop_event):
    """
    Read frames from a video file and add them to frames_dict.

    Args:
        file_path (str): Path to the video file.
        frames_dict (dict): Dictionary to store the read video frames.
        stop_event (threading.Event): Event to stop the thread.
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
    Calculate the similarity between two feature vectors.

    Args:
        feat1: First feature vector.
        feat2: Second feature vector.
        size1: Size (area) of the first person.
        size2: Size (area) of the second person.

    Returns:
        float: Similarity score.
    """
    epsilon = 1e-8  # Small value to avoid division by zero

    # Calculate cosine similarity
    feat1_norm = np.linalg.norm(feat1)
    feat2_norm = np.linalg.norm(feat2)
    if feat1_norm == 0 or feat2_norm == 0:
        cos_sim = 0.0
    else:
        cos_sim = np.dot(feat1, feat2.T) / (feat1_norm * feat2_norm)
        cos_sim = cos_sim.item()  # Convert to scalar value

    # Calculate size similarity
    size_sim = 1 - np.abs(size1 - size2) / (np.maximum(size1, size2) + epsilon)

    # Combine similarity scores
    similarity = 0.7 * cos_sim + 0.3 * size_sim

    return similarity


def draw_rectangle(frame, person, color, object_id):
    """
    Draw a rectangle and object number on the frame.

    Args:
        frame: Frame to draw the rectangle on.
        person: Coordinate information of the person to draw the rectangle for.
        color: Color of the rectangle.
        object_id: Number of the object.
    """
    x1, y1, x2, y2 = map(int, person)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, str(object_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def get_scale(frame_width, frame_height, orig_shape):
    """
    Calculate the scale ratios based on the frame size and original image size.

    Args:
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.
        orig_shape (tuple): Size of the original image (height, width).

    Returns:
        tuple: Scale ratios for width and height (scale_x, scale_y).
    """
    orig_height, orig_width = orig_shape[:2]
    scale_x = frame_width / orig_width
    scale_y = frame_height / orig_height
    return scale_x, scale_y


def extract_features(track_results):
    """
    Extract appearance features for each person from the detected results.

    Args:
        track_results (list): Detection results from the YOLOv8 model.

    Returns:
        tuple: (features_list, sizes_list, max_feature_length)
            - features_list (list): List of appearance features for each person in each frame.
            - sizes_list (list): List of sizes for each person in each frame.
            - max_feature_length (int): Length of the longest feature vector.
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
    Match persons between frames based on appearance features using the Hungarian algorithm.

    Args:
        padded_features_list (list): List of padded appearance features.
        sizes_list (list): List of sizes for each person in each frame.

    Returns:
        list: Matched index information.
    """
    num_frames = len(padded_features_list)
    matched_indices = [[] for _ in range(num_frames)]

    for i in range(num_frames - 1):
        cos_sim = np.zeros((len(padded_features_list[i]), len(padded_features_list[i + 1])))
        for j in range(len(padded_features_list[i])):
            feat1 = np.array(padded_features_list[i][j]).reshape(1, -1)
            for k in range(len(padded_features_list[i + 1])):
                feat2 = np.array(padded_features_list[i + 1][k]).reshape(1, -1)
                size1 = sizes_list[i][j]
                size2 = sizes_list[i + 1][k]
                cos_sim[j, k] = calculate_similarity(feat1, feat2, size1, size2)

        # Convert the similarity matrix to a minimum assignment problem by taking the negative
        row_ind, col_ind = linear_sum_assignment(-cos_sim)
        matched_indices[i] = col_ind.tolist()

    return matched_indices


def visualize_results(track_results, matched_indices):
    """
    Visualize the matching results on the frames.

    Args:
        track_results (list): Detection results from the YOLOv8 model.
        matched_indices (list): Matched index information.
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

            if i < len(matched_indices) and j < len(matched_indices[i]):
                match_idx = matched_indices[i][j]
                if match_idx < len(persons):
                    match_person = persons[match_idx]
                    scale_x_match, scale_y_match = get_scale(frame_width, frame_height,
                                                             track_results[i + 1][0].orig_img.shape)
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
    Print the matching information.

    Args:
        matched_indices (list): Matched index information.
    """
    num_frames = len(matched_indices)

    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            for k, match_idx in enumerate(matched_indices[i][j]):
                print(f'Person {k + 1} in frame {i + 1} is the same as person {match_idx + 1} in frame {j + 1}')

    print('---')


def main():
    frames_dict = {file_path: [] for file_path in config.VIDEO_FILES}
    stop_event = threading.Event()

    # Create and start threads for each video file
    threads = []
    for i, file_path in enumerate(config.VIDEO_FILES):
        t = threading.Thread(target=video_reader, args=(file_path, frames_dict, stop_event))
        threads.append(t)
        t.start()

    frame_count = 0
    while True:
        if all(len(frames) > 0 for frames in frames_dict.values()):
            # Extract frames from frames_dict
            frames = [frames_dict[file_path].pop(0) for file_path in config.VIDEO_FILES]

            if frame_count % config.KEYFRAME_INTERVAL == 0:
                # Perform object detection using YOLOv8 for each frame
                track_results = [model(frame, device=0, classes=0, conf=0.6) for frame in frames]

                # Extract appearance features for each person from the detected results
                features_list, sizes_list, max_feature_length = extract_features(track_results)

                # Unify the dimensions of the feature vectors
                padded_features_list = []
                for features in features_list:
                    padded_features = []
                    for feature in features:
                        padded_feature = feature + [0] * (max_feature_length - len(feature))
                        padded_features.append(padded_feature)
                    padded_features_list.append(padded_features)

                # Match persons between frames based on appearance features
                matched_indices = match_persons(padded_features_list, sizes_list)

                # Visualize the matching results on the frames
                visualize_results(track_results, matched_indices)

            frame_count += 1

        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # Wait for all threads to finish
    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
