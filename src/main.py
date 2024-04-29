import colorsys
import threading

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

import config

EPSILON = 1e-8  # Small value to avoid division by zero

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


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Args:
        bbox1 (list): First bounding box coordinates [x1, y1, x2, y2].
        bbox2 (list): Second bounding box coordinates [x1, y1, x2, y2].
    Returns:
        float: IoU value.
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0  # Return 0 if the bounding box coordinates are invalid

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / (union_area + EPSILON)
    return iou


def calculate_similarity(feat1, feat2, size1, size2, color_hist1, color_hist2):
    """
    Calculate the similarity between two feature vectors.
    Args:
        feat1: First feature vector.
        feat2: Second feature vector.
        size1: Size (area) of the first person.
        size2: Size (area) of the second person.
        color_hist1: Color histogram of the first person.
        color_hist2: Color histogram of the second person.
    Returns:
        float: Similarity score.
    """
    # Ensure feature vectors have the same length
    min_length = min(feat1.shape[1], feat2.shape[1])
    feat1 = feat1[:, :min_length]
    feat2 = feat2[:, :min_length]

    # Calculate cosine similarity
    feat1_norm = np.linalg.norm(feat1)
    feat2_norm = np.linalg.norm(feat2)
    if feat1_norm > 0 and feat2_norm > 0:
        cos_sim = np.dot(feat1, feat2.T) / (feat1_norm * feat2_norm)
        cos_sim = cos_sim.item()  # Convert to scalar value
    else:
        cos_sim = 0.0

    # Calculate size similarity
    size_sim = 1 - np.abs(size1 - size2) / (max(size1, size2) + EPSILON)

    # Calculate color histogram similarity
    if color_hist1 is not None and color_hist2 is not None:
        color_hist1 = np.array(color_hist1)
        color_hist2 = np.array(color_hist2)
        color_sim = np.dot(color_hist1, color_hist2.T)
    else:
        color_sim = 0.0

    # Combine similarity scores
    similarity = 0.5 * cos_sim + 0.3 * size_sim + 0.2 * color_sim

    return similarity


def calculate_sort_similarity(features_list, i, j):
    """
    Calculate similarity matrix for SORT algorithm.
    Args:
        features_list (list): List of appearance features.
        i (int): Index of the first frame.
        j (int): Index of the second frame.
    Returns:
        numpy.ndarray: Similarity matrix.
    """
    iou_matrix = np.zeros((len(features_list[i]), len(features_list[j])))
    for k in range(len(features_list[i])):
        for L in range(len(features_list[j])):
            iou_matrix[k, L] = calculate_iou(features_list[i][k], features_list[j][L])
    return iou_matrix


def calculate_deepsort_similarity(features_list, sizes_list, color_hists_list, i, j):
    """
    Calculate similarity matrix for DeepSORT algorithm.
    Args:
        features_list (list): List of appearance features.
        sizes_list (list): List of sizes for each person in each frame.
        color_hists_list (list): List of color histograms for each person in each frame.
        i (int): Index of the first frame.
        j (int): Index of the second frame.
    Returns:
        numpy.ndarray: Similarity matrix.
    """
    cos_sim = np.zeros((len(features_list[i]), len(features_list[j])))
    for k in range(len(features_list[i])):
        feat1 = np.array(features_list[i][k]).reshape(1, -1)
        for L in range(len(features_list[j])):
            feat2 = np.array(features_list[j][L]).reshape(1, -1)
            size1 = sizes_list[i][k]
            size2 = sizes_list[j][L]
            if k < len(color_hists_list[i]) and L < len(color_hists_list[j]):
                color_hist1 = color_hists_list[i][k]
                color_hist2 = color_hists_list[j][L]
                cos_sim[k, L] = calculate_similarity(feat1, feat2, size1, size2, color_hist1, color_hist2)
            else:
                cos_sim[k, L] = calculate_similarity(feat1, feat2, size1, size2, None, None)
    return cos_sim


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


def generate_colors(num_colors):
    """
    Generate a list of distinct colors.
    Args:
        num_colors (int): Number of colors to generate.
    Returns:
        list: List of RGB color tuples.
    """
    colors = []
    hue_step = 1.0 / num_colors
    for i in range(num_colors):
        hue = i * hue_step
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def extract_features(track_results):
    """
    Extract appearance features for each person from the detected results.
    Args:
        track_results (list): Detection results from the YOLOv8 model.
    Returns:
        tuple: (features_list, sizes_list, color_hists_list)
            - features_list (list): List of appearance features for each person in each frame.
            - sizes_list (list): List of sizes for each person in each frame.
            - color_hists_list (list): List of color histograms for each person in each frame.
    """
    features_list = []
    sizes_list = []
    color_hists_list = []

    for frame_results in track_results:
        features = []
        sizes = []
        color_hists = []
        for result in frame_results:
            for person in result.boxes.data.tolist():
                x1, y1, x2, y2 = map(int, person[:4])
                person_img = result.orig_img[y1:y2, x1:x2]
                person_result = model(person_img)
                feature = person_result[0].boxes.conf.tolist()
                features.append(feature)
                sizes.append((x2 - x1) * (y2 - y1))

                # Calculate color histogram
                color_hist, _ = np.histogram(person_img.reshape(-1, 3), bins=8, range=(0, 256))
                color_hist = color_hist / np.sum(color_hist)
                color_hists.append(color_hist)

        features_list.append(features)
        sizes_list.append(sizes)
        color_hists_list.append(color_hists)

    return features_list, sizes_list, color_hists_list


def match_persons(features_list, sizes_list, color_hists_list, use_sort_ratio, prev_frame_data, occluded_tracks,
                  base_frame_index):
    """
    Match persons between frames based on appearance features using the Hungarian algorithm.
    Args:
        features_list (list): List of appearance features.
        sizes_list (list): List of sizes for each person in each frame.
        color_hists_list (list): List of color histograms for each person in each frame.
        use_sort_ratio (float): Ratio of frames to apply SORT algorithm instead of DeepSORT.
        prev_frame_data (dict): Data from the previous frame.
        occluded_tracks (dict): Dictionary to store occluded tracks.
        base_frame_index (int): Index of the base frame.
    Returns:
        tuple: (matched_indices, curr_frame_data, occluded_tracks)
            - matched_indices (list): Matched index information.
            - curr_frame_data (dict): Data for the current frame.
            - occluded_tracks (dict): Updated dictionary of occluded tracks.
    """
    num_frames = len(features_list)
    matched_indices = [[[] for _ in range(num_frames)] for _ in range(num_frames)]
    curr_frame_data = {}

    for i in range(num_frames):
        curr_frame_data[i] = {}
        if i != base_frame_index:
            if np.random.rand() < use_sort_ratio:
                # Apply SORT algorithm
                similarity_matrix = calculate_sort_similarity(features_list, base_frame_index, i)
            else:
                # Apply DeepSORT algorithm
                similarity_matrix = calculate_deepsort_similarity(features_list, sizes_list, color_hists_list,
                                                                  base_frame_index, i)

            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

            matched_indices[base_frame_index][i] = col_ind.tolist()
            matched_indices[i][base_frame_index] = row_ind.tolist()

            # Update Kalman filter for matched objects
            for k, L in zip(row_ind, col_ind):
                feature_length = len(features_list[i][L])
                kf = prev_frame_data.get((base_frame_index, k),
                                         KalmanFilter(dim_x=feature_length, dim_z=feature_length))

                kf.F = np.eye(feature_length)
                kf.H = np.eye(feature_length)
                kf.R = np.eye(feature_length) * 10.0
                kf.P = np.eye(feature_length) * 1000.0
                kf.Q = np.eye(feature_length) * 0.01
                kf.predict()
                kf.update(np.array(features_list[i][L]).reshape(-1, 1))

                curr_frame_data[i][L] = kf

            # Handle occluded tracks
            for k in range(len(features_list[base_frame_index])):
                if k not in row_ind:
                    if (base_frame_index, k) in prev_frame_data:
                        kf = prev_frame_data[(base_frame_index, k)]
                        kf.predict()
                        occluded_tracks[(base_frame_index, k)] = kf
                    else:
                        occluded_tracks[(base_frame_index, k)] = None

    # Re-identify occluded tracks
    for (i, k), kf in occluded_tracks.items():
        if kf is not None:
            best_match = None
            best_similarity = -1
            for j in range(num_frames):
                if j != base_frame_index:
                    for L in range(len(features_list[j])):
                        if (j, L) not in curr_frame_data.values():
                            feat = np.array(features_list[j][L]).reshape(1, -1)
                            similarity = calculate_similarity(kf.x, feat, sizes_list[base_frame_index][k],
                                                              sizes_list[j][L],
                                                              color_hists_list[base_frame_index][k],
                                                              color_hists_list[j][L])
                            if similarity > best_similarity:
                                best_match = (j, L)
                                best_similarity = similarity
            if best_match is not None and best_similarity > config.REID_THRESHOLD:
                curr_frame_data[best_match[0]][best_match[1]] = kf
                matched_indices[base_frame_index][best_match[0]].append(best_match[1])
                matched_indices[best_match[0]][base_frame_index].append(k)

    return matched_indices, curr_frame_data, occluded_tracks


def visualize_results(track_results, matched_indices, base_frame_index):
    """
    Visualize the matching results on the frames.
    Args:
        track_results (list): Detection results from the YOLOv8 model.
        matched_indices (list): Matched index information.
        base_frame_index (int): Index of the base frame.
    """
    frame_width = config.FRAME_SIZE['w']
    frame_height = config.FRAME_SIZE['h']

    max_people = max(len(result[0].boxes.data.tolist()) for result in track_results)
    colors = generate_colors(max_people)

    for i in range(len(track_results)):
        frame = cv2.resize(track_results[i][0].orig_img, (frame_width, frame_height))

        if i == base_frame_index:
            cv2.circle(frame, (10, 10), 5, (0, 0, 255), -1)

        persons = track_results[i][0].boxes.data.tolist()

        for j, person in enumerate(persons):
            scale_x, scale_y = get_scale(frame_width, frame_height, track_results[i][0].orig_img.shape)
            person_coords = [person[0] * scale_x, person[1] * scale_y, person[2] * scale_x, person[3] * scale_y]

            if i == base_frame_index:
                color = colors[j % len(colors)]
                object_id = j + 1
            else:
                base_match_indices = matched_indices[base_frame_index][i]
                if j < len(base_match_indices):
                    base_match_idx = base_match_indices[j]
                    color = colors[base_match_idx % len(colors)]
                    object_id = base_match_idx + 1
                else:
                    color = (255, 255, 255)  # Non-matched people are colored white
                    object_id = None

            draw_rectangle(frame, person_coords, color, object_id)

        cv2.imshow(f'Video {i + 1}', frame)


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
    use_sort_ratio = config.USE_SORT_RATIO  # Ratio of frames to apply SORT algorithm
    prev_frame_data = {}
    occluded_tracks = {}

    while True:
        if all(len(frames) > 0 for frames in frames_dict.values()):
            # Extract frames from frames_dict
            frames = [frames_dict[file_path].pop(0) for file_path in config.VIDEO_FILES]

            if frame_count % config.KEYFRAME_INTERVAL == 0:
                # Perform object detection using YOLOv8 for each frame
                track_results = [model(frame, device=0, classes=0, conf=0.6) for frame in frames]

                # Extract appearance features for each person from the detected results
                features_list, sizes_list, color_hists_list = extract_features(track_results)

                # Find the frame with the most detected people
                num_people_per_frame = [len(features) for features in features_list]
                base_frame_index = num_people_per_frame.index(max(num_people_per_frame))

                matched_indices, curr_frame_data, occluded_tracks = match_persons(features_list, sizes_list,
                                                                                  color_hists_list,
                                                                                  use_sort_ratio, prev_frame_data,
                                                                                  occluded_tracks, base_frame_index)
                prev_frame_data = curr_frame_data

                # Visualize the matching results on the frames
                visualize_results(track_results, matched_indices, base_frame_index)

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
