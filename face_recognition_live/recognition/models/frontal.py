from enum import Enum
from collections import namedtuple


from face_recognition_live.recognition.models.landmarks import DLIB68_FACE_LOCATIONS


class FRONTAL_MODEL(Enum):
    NOSE_LOCATION = 1


FrontalResult = namedtuple("FrontalResult", ["is_frontal", "frontal_box"])


NOSE_CENTERED_TOLERANCE_X = 0.05
NOSE_CENTERED_TOLERANCE_Y = 0.05


def nose_centered(bounding_box, landmarks_68):
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

    return FrontalResult(is_centered, (acceptable_nose_location_left, acceptable_nose_location_right,
                                       acceptable_nose_location_top, acceptable_nose_location_bottom))


def init_frontal_detection(model_type):
    if model_type == FRONTAL_MODEL.NOSE_LOCATION:
        frontal_model = nose_centered
    else:
        raise NotImplementedError()

    def run(bounding_box, landmarks):
        landmarks = landmarks.parts()
        assert len(landmarks) == 68
        return frontal_model(bounding_box, landmarks)

    return run

