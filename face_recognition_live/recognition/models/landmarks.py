from pathlib import Path
from enum import Enum

import dlib


class LANDMARK_MODEL(Enum):
    DLIB_68 = 1
    DLIB_5 = 2


class DLIB68_FACE_LOCATIONS(Enum):
    CHIN = 8
    NOSE_TIP = 30
    NOSE_BOTTOM = 33
    LEFT_EYE_LEFT_CORNER = 36
    RIGHT_EYE_RIGHT_CORNER = 45
    MOUTH_LEFT = 48
    MOUTH_RIGHT = 54


def init_landmark_detection(model_type, model_dir: Path):

    if model_type == LANDMARK_MODEL.DLIB_68:
        model_location = model_dir / "landmark_detection" / "shape_predictor_68_face_landmarks.dat"
        landmark_model = dlib.shape_predictor(str(model_location))
    elif model_type == LANDMARK_MODEL.DLIB_68:
        model_location = model_dir / "landmark_detection" / "shape_predictor_5_face_landmarks.dat"
        landmark_model = dlib.shape_predictor(str(model_location))
    else:
        raise NotImplementedError()

    def run(rgb_image, bounding_box):
        return landmark_model(rgb_image, bounding_box)

    return run
