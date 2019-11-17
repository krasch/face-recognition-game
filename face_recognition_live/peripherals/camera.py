from contextlib import contextmanager
import threading
from queue import Queue
import time

import cv2

from face_recognition_live.events.results import CameraImage
from face_recognition_live.config import CONFIG


def init_jetson_video_capture():
    capture_width = CONFIG["source_settings"]["camera_jetson"]["width"]
    capture_height = CONFIG["source_settings"]["camera_jetson"]["height"]
    flip = CONFIG["source_settings"]["camera_jetson"]["flip"]
    zoom = CONFIG["source_settings"]["camera_jetson"]["zoom"]
    framerate = CONFIG["source_settings"]["camera_jetson"]["framerate"]

    display_width = CONFIG["source_settings"]["camera_jetson"]["display_width"]
    display_height = CONFIG["source_settings"]["camera_jetson"]["display_height"]

    # calculate crop window
    # somewhat unintuitively gstreamer calculates it such that the cropped window still has capture_height*capture_width
    assert zoom >= 1
    crop_left = int((capture_width - capture_width/zoom)/2.0)
    crop_right = capture_width - crop_left
    crop_top = int((capture_height - capture_height/zoom)/2.0)
    crop_bottom = capture_height - crop_top

    gstreamer_config = (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            f'nvvidconv flip-method={flip} ! ' +
            #f'nvvidconv flip-method={flip} left={crop_left} right={crop_right} top={crop_top} bottom={crop_bottom} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

    return cv2.VideoCapture(gstreamer_config, cv2.CAP_GSTREAMER)


def init_video_capture():
    return cv2.VideoCapture(CONFIG["source_settings"]["camera"]["location"])


def init_file_streaming():
    class SlowerStream:
        def __init__(self, fast_stream, frames_per_second):
            self.fast_stream = fast_stream
            self.framerate = 1.0 / frames_per_second

        def read(self):
            time.sleep(self.framerate)
            has_image, image = self.fast_stream.read()
            return has_image, image

        def release(self):
            self.fast_stream.release()

    return SlowerStream(cv2.VideoCapture(CONFIG["source_settings"]["prerecorded"]["location"]), 5)


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


@contextmanager
def open_stream():
    if CONFIG["source"] == "camera_jetson":
        video_capture = init_jetson_video_capture()
    elif CONFIG["source"] == "camera":
        video_capture = init_video_capture()
    elif CONFIG["source"] == "prerecorded":
        video_capture = init_file_streaming()
    else:
        raise NotImplementedError()

    try:
        yield video_capture
    finally:
        video_capture.release()


@contextmanager
def init_camera():
    def read_from_camera(camera):
        counter = 0
        while True:
            has_image, image = camera.read()
            if has_image:
                image = increase_brightness(image, CONFIG["source_settings"]["increase_brightness"])
                if CONFIG["source_settings"]["mirror"]:
                    image = cv2.flip(image, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield CameraImage(counter, image)
                counter = (counter + 1) % 10000

    with open_stream() as stream:
        yield read_from_camera(stream)

