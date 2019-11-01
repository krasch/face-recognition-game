from pathlib import Path
from collections import namedtuple

from face_recognition_live.recognition.models.detection import init_face_detector, DETECTION_MODEL
from face_recognition_live.recognition.models.landmarks import init_landmark_detection, LANDMARK_MODEL
from face_recognition_live.recognition.models.frontal import init_frontal_detection, FRONTAL_MODEL
from face_recognition_live.recognition.models.cropping import init_face_cropper, CROPPING_METHOD
from face_recognition_live.recognition.models.features import init_feature_extractor, FEATURE_EXTRACTION_MODEL


ModelStack = namedtuple("ModelStack", ["detect_faces", "find_landmarks", "is_frontal", "crop", "extract_features"])


def init_model_stack(config):
    if config["recognition"]["models"]["stack"] == "CNN+dlibcrop+openface":
        detection_model = DETECTION_MODEL.CNN
        landmark_model = LANDMARK_MODEL.DLIB_68
        frontal_model = FRONTAL_MODEL.NOSE_LOCATION
        cropping_method = CROPPING_METHOD.DLIB_68
        feature_model = FEATURE_EXTRACTION_MODEL.OPEN_FACE_NN4_SMALL2

    elif config["recognition"]["models"]["stack"] == "HOG+dlibcrop+openface":
        detection_model = DETECTION_MODEL.HOG
        landmark_model = LANDMARK_MODEL.DLIB_68
        frontal_model = FRONTAL_MODEL.NOSE_LOCATION
        cropping_method = CROPPING_METHOD.DLIB_68
        feature_model = FEATURE_EXTRACTION_MODEL.OPEN_FACE_NN4_SMALL2
    else:
        raise NotImplementedError()

    model_dir = Path(config["recognition"]["models"]["directory"])
    scale = config["recognition"]["models"]["face_detection"]["scale"]

    return ModelStack(detect_faces=init_face_detector(detection_model, model_dir, scale),
                      find_landmarks=init_landmark_detection(landmark_model, model_dir),
                      is_frontal=init_frontal_detection(frontal_model),
                      crop=init_face_cropper(cropping_method),
                      extract_features=init_feature_extractor(feature_model, model_dir))
