from unittest import mock
from face_recognition_live.models.face_detection import init_face_detector


def test_face_partially_outside_camera_window():
    config = mock.Mock()
    init_face_detector(config)
    assert False
    #config = {"recognition": {"models": {"directory": "models"},
    #                          "face_detection": {"model": "cnn"}}}
    #detector = init_face_detector("cnn")


