from pathlib import Path

from face_recognition_live.models.face_detection import init_face_detector
from face_recognition_live.models.face_cropping import init_face_cropper
from face_recognition_live.models.feature_extraction import init_feature_extractor


def init_model_stack(config):
    model_dir = Path(config["recognition"]["models"]["directory"])

    if config["recognition"]["models"]["stack"] == "CNN+dlibcrop+openface":
        scale = config["recognition"]["models"]["face_detection"]["scale"]
        face_detector = init_face_detector(model_dir, "CNN", scale)
        face_cropper = init_face_cropper(model_dir)
        feature_extractor = init_feature_extractor(model_dir)
        return face_detector, face_cropper, feature_extractor
    elif config["recognition"]["models"]["stack"] == "HOG+dlibcrop+openface":
        scale = config["recognition"]["models"]["face_detection"]["scale"]
        face_detector = init_face_detector(model_dir, "HOG", scale)
        face_cropper = init_face_cropper(model_dir)
        feature_extractor = init_feature_extractor(model_dir)
        return face_detector, face_cropper, feature_extractor
    else:
        raise NotImplementedError()
