from pathlib import Path

import cv2

from face_recognition_live.models.face_detection import init_face_detector
from tests.util import assert_face_detections_equal


# todo avoid loading models
def test_results_scale_invariant():
    image = cv2.imread("tests/data/1.png")

    detector_fullsize = init_face_detector(Path("models"), "CNN", 1.0)
    detector_halfsize = init_face_detector(Path("models"), "CNN", 0.5)

    faces_fullsize = detector_fullsize(image)
    faces_halfsize = detector_halfsize(image)

    assert_face_detections_equal(faces_fullsize, faces_halfsize)

# todo test not returning faces outside image
