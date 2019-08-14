from pathlib import Path

import dlib


def init_face_cropper(model_dir: Path):
    model_location = model_dir / "face_cropping" / "shape_predictor_5_face_landmarks.dat"

    landmark_model = dlib.shape_predictor(str(model_location))

    def run(frame, faces):
        shapes = [landmark_model(frame[:, :, ::-1], f) for f in faces]
        return [dlib.get_face_chip(frame, s)[:, :, ::-1] for s in shapes]  # todo 96

    return run
