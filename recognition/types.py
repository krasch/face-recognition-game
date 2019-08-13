from enum import Enum


class ResultTypes(Enum):
    CAMERA_IMAGE = 0
    FACE_DETECTIONS = 1


class Result:
    def __init__(self, type_: ResultTypes, data=None):
        self.type = type_
        self.data = data


class TaskTypes(Enum):
    DETECT_FACES = 0
    RECOGNIZE_PERSON = 1


class Task:
    def __init__(self, type_: TaskTypes, data=None):
        self.type = type_
        self.data = data

