from pathlib import Path

import cv2


def init_feature_extractor(model_dir: Path):
    model_location = model_dir / "feature_extraction" / "nn4.small2.v1.t7"

    feature_model = cv2.dnn.readNetFromTorch(str(model_location))

    def run(faces):
        blob = cv2.dnn.blobFromImages(faces, 1.0 / 255, (96, 96), swapRB=False, crop=False)  # todo resize should not be necessary
        feature_model.setInput(blob)
        return feature_model.forward()

    return run
