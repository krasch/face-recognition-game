from enum import Enum

import numpy as np
import cv2

from face_recognition_live.models.landmarks import DLIB68_FACE_LOCATIONS



class FRONTAL_MODEL(Enum):
    NOSE_LOCATION = 1
    HEAD_POSE = 2


NOSE_CENTERED_TOLERANCE_X = 0.05
NOSE_CENTERED_TOLERANCE_Y = 0.05


def nose_centered(image, bounding_box, landmarks_68):
    box_height = bounding_box.bottom() - bounding_box.top()
    box_width = bounding_box.right() - bounding_box.left()

    expected_nose_location_x = bounding_box.left() + int(box_width / 2.0)
    expected_nose_location_y = bounding_box.top() + int(box_height / 2.0)

    diff_x = int(box_width * NOSE_CENTERED_TOLERANCE_X)
    diff_y = int(box_height * NOSE_CENTERED_TOLERANCE_Y)

    acceptable_nose_location_left = expected_nose_location_x - diff_x
    acceptable_nose_location_right = expected_nose_location_x + diff_x
    acceptable_nose_location_top = expected_nose_location_y - diff_y
    acceptable_nose_location_bottom = expected_nose_location_y + diff_y

    actual_nose_location_x = landmarks_68[DLIB68_FACE_LOCATIONS.NOSE_TIP.value].x
    actual_nose_location_y = landmarks_68[DLIB68_FACE_LOCATIONS.NOSE_TIP.value].y

    is_centered = (acceptable_nose_location_left <= actual_nose_location_x <= acceptable_nose_location_right) & \
                  (acceptable_nose_location_top  <= actual_nose_location_y <= acceptable_nose_location_bottom)

    return is_centered, (acceptable_nose_location_left,
                         acceptable_nose_location_right,
                         acceptable_nose_location_top,
                         acceptable_nose_location_bottom)


def head_pose(image, landmarks_68):
    pass


def init_frontal_detection(model_type):
    if model_type == FRONTAL_MODEL.NOSE_LOCATION:
        frontal_model = nose_centered
    elif model_type == FRONTAL_MODEL.HEAD_POSE:
        frontal_model = head_pose
    else:
        raise NotImplementedError()

    def run(image, bounding_box, landmarks):
        landmarks = landmarks.parts()
        assert len(landmarks) == 68
        return frontal_model(image, bounding_box, landmarks)

    return run

""""
FACE_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corne
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner

])


def extract_pose_estimation_landmarks(landmarks):
    return np.array([(landmarks[33].x, landmarks[33].y),  # Nose tip   # todo
                     (landmarks[8].x, landmarks[8].y),    # Chin
                     (landmarks[36].x, landmarks[36].y),  # Left eye left corner
                     (landmarks[45].x, landmarks[45].y),  # Right eye right corne
                     (landmarks[48].x, landmarks[48].y),  # Left Mouth corner
                     (landmarks[54].x, landmarks[54].y)   # Right mouth corner
                     ], dtype="double")


def calculate_camera_matrix(image):
    # todo calibrate
    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    return camera_matrix


# https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
def calculate_arrow(camera_matrix, landmarks):
    landmarks = extract_pose_estimation_landmarks(landmarks)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(FACE_POINTS, landmarks, camera_matrix,
                                                                dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                                   rotation_vector, translation_vector,
                                                   camera_matrix, dist_coeffs)

    return {"nose": (int(landmarks[0][0]), int(landmarks[0][1])),
            "chin": (int(landmarks[1][0]), int(landmarks[1][1])),
            "eye_left": (int(landmarks[2][0]), int(landmarks[2][1])),
            "eye_right": (int(landmarks[3][0]), int(landmarks[3][1])),
            "mouth_left": (int(landmarks[4][0]), int(landmarks[4][1])),
            "mouth_right": (int(landmarks[5][0]), int(landmarks[5][1])),
            "arrow_end": (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))}


def init_frontal_detector():

    def run(image, landmarks):
        camera_matrix = calculate_camera_matrix(image)
        return calculate_arrow(camera_matrix, landmarks)

    return run
"""


