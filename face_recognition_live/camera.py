from contextlib import contextmanager
import platform
import threading
from queue import Queue
import time

import cv2

from face_recognition_live.results import CameraImage


def get_jetson_gstreamer_source(config):
    capture_width = config["camera"]["width"]
    capture_height = config["camera"]["weight"]
    display_width = config["display"]["width"]
    display_height = config["display"]["height"]
    flip = config["camera"]["flip"]
    zoom = config["camera"]["zoom"]
    framerate = config["camera"]["framerate"]

    # calculate crop window
    # somewhat unintuitively gstreamer calculates it such that the cropped window still has capture_height*capture_width
    assert config[""] >= 1
    crop_left = int((capture_width - capture_width/zoom)/2.0)
    crop_right = capture_width - crop_left
    crop_top = int((capture_height - capture_height/zoom)/2.0)
    crop_bottom = capture_height - crop_top

    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){capture_width}, height=(int){capture_height}, ' +
            f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            # f'nvvidconv flip-method={flip_method} ! ' +
            f'nvvidconv flip-method={flip} left={crop_left} right={crop_right} top={crop_top} bottom={crop_bottom} ! ' +
            f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )


def running_on_jetson():
    return platform.machine() == "aarch64"


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


@contextmanager
def open_stream(config):
    if "prerecorded_frames" in config:
        video_capture = SlowerStream(cv2.VideoCapture(config["prerecorded_frames"]), 15)
    elif running_on_jetson():
        video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(config), cv2.CAP_GSTREAMER)
    else:
        video_capture = cv2.VideoCapture(0)

    try:
        yield video_capture
    finally:
        video_capture.release()


class CameraThread(threading.Thread):
    COUNTER_WRAPAROUND = 10000

    def __init__(self, config, results_queue: Queue):
        super(CameraThread, self).__init__()
        self.results_queue = results_queue
        self.stoprequest = threading.Event()
        self.config = config

    def run(self):
        with open_stream(self.config) as stream:
            counter = 0
            while not self.stoprequest.isSet():
                has_image, image = stream.read()
                if has_image:
                    self.results_queue.put(CameraImage(counter, image))
                    counter = (counter + 1) % self.COUNTER_WRAPAROUND

    def join(self, timeout=None):
        self.stoprequest.set()
        super(CameraThread, self).join(timeout)


@contextmanager
def init_camera(config, results_queue):
    camera = CameraThread(config, results_queue)

    try:
        camera.start()
        yield camera
    finally:
        camera.join()
