from pathlib import Path

import dlib
import numpy as np

"""
def angle(landmarks):
    #left_x = (landmarks[2].x() + landmarks[3].x)/2.0
    right_right, right_left, left_left, left_right, nose = landmarks

    dY = right_right.y() - left_left.y()
    dX = right_right.x() - left_left.x()
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    print(a)
"""


def init_face_cropper(model_dir: Path):
    model_location = model_dir / "face_cropping" / "shape_predictor_5_face_landmarks.dat"

    landmark_model = dlib.shape_predictor(str(model_location))  # todo follow openface alignment

    def run(image, faces):
        shapes = [landmark_model(image[:, :, ::-1], f) for f in faces]
        return [dlib.get_face_chip(image, s) for s in shapes]  # todo 96

    return run
