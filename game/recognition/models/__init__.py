from pathlib import Path
from collections import namedtuple

from game.config import CONFIG
from game.recognition.models.detection import init_face_detector, DETECTION_MODEL
from game.recognition.models.landmarks import init_landmark_detection, LANDMARK_MODEL
from game.recognition.models.cropping import init_face_cropper, CROPPING_METHOD
from game.recognition.models.features import init_feature_extractor, FEATURE_EXTRACTION_MODEL
from game.recognition.models.matching import init_matching, MATCHING_METHOD


ModelStack = namedtuple("ModelStack", ["detect_faces", "find_landmarks", "crop", "extract_features", "match_faces"])


def init_model_stack():
    if CONFIG["recognition"]["models"]["stack"] == "CNN+dlibcrop+openface+dot":
        detection_model = DETECTION_MODEL.CNN
        landmark_model = LANDMARK_MODEL.DLIB_68
        cropping_method = CROPPING_METHOD.OPEN_FACE_DLIB_BASED
        feature_model = FEATURE_EXTRACTION_MODEL.OPEN_FACE_NN4_SMALL2
        matching_method = MATCHING_METHOD.DOT

    elif CONFIG["recognition"]["models"]["stack"] == "HOG+dlibcrop+openface+dot":
        detection_model = DETECTION_MODEL.HOG
        landmark_model = LANDMARK_MODEL.DLIB_68
        cropping_method = CROPPING_METHOD.OPEN_FACE_DLIB_BASED
        feature_model = FEATURE_EXTRACTION_MODEL.OPEN_FACE_NN4_SMALL2
        matching_method = MATCHING_METHOD.DOT
    else:
        raise NotImplementedError()

    model_dir = Path(CONFIG["recognition"]["models"]["directory"])

    return ModelStack(detect_faces=init_face_detector(detection_model, model_dir),
                      find_landmarks=init_landmark_detection(landmark_model, model_dir),
                      crop=init_face_cropper(cropping_method),
                      extract_features=init_feature_extractor(feature_model, model_dir),
                      match_faces=init_matching(matching_method))
