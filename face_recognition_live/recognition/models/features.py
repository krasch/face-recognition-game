from pathlib import Path
from enum import Enum

import cv2


class FEATURE_EXTRACTION_MODEL(Enum):
    OPEN_FACE_NN4_SMALL2 = 1


def init_feature_extractor(model_type, model_dir: Path):
    if model_type != FEATURE_EXTRACTION_MODEL.OPEN_FACE_NN4_SMALL2:
        raise NotImplementedError()

    model_location = model_dir / "feature_extraction" / "nn4.small2.v1.t7"
    feature_model = cv2.dnn.readNetFromTorch(str(model_location))

    def run(faces):
        # no faces found
        if len(faces.shape) == 1:
            return []

        blob = cv2.dnn.blobFromImages(faces, 1.0 / 255, (96, 96), swapRB=False, crop=False)  # todo resize should not be necessary
        feature_model.setInput(blob)
        return feature_model.forward()

    return run
