from enum import Enum

import dlib


class CROPPING_METHOD(Enum):
    DLIB_68 = 1


def init_face_cropper(cropping_method):
    if cropping_method != CROPPING_METHOD.DLIB_68:
        raise NotImplementedError()

    def run(image, landmarks):
        return dlib.get_face_chip(image, landmarks) # todo 96

    return run
