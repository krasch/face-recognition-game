from contextlib import contextmanager
import platform
import threading
from queue import Queue

import cv2

from recognition.results import CameraImage

# CAMERA SETTINGS
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FLIP = 0
ZOOM = 1.00
FRAMERATE = 15

# SCREEN SETTINGS
# todo check screen ratio when fullscreen selected
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def get_jetson_gstreamer_source():
    # calculate crop window
    # somewhat unintuitively gstreamer will calculate it such that the cropped window still has CAPTURE_HEIGHT*CAPTURE_WIDTH
    assert ZOOM >= 1
    crop_left = int((CAPTURE_WIDTH - CAPTURE_WIDTH/ZOOM)/2.0)
    crop_right = CAPTURE_WIDTH - crop_left
    crop_top = int((CAPTURE_HEIGHT - CAPTURE_HEIGHT/ZOOM)/2.0)
    crop_bottom = CAPTURE_HEIGHT - crop_top

    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}, ' +
            f'format=(string)NV12, framerate=(fraction){FRAMERATE}/1 ! ' +
            # f'nvvidconv flip-method={flip_method} ! ' +
            f'nvvidconv flip-method={FLIP} left={crop_left} right={crop_right} top={crop_top} bottom={crop_bottom} ! ' +
            f'video/x-raw, width=(int){DISPLAY_WIDTH}, height=(int){DISPLAY_HEIGHT}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )


@contextmanager
def open_stream():
    if platform.machine() == "aarch64":
        video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        # video_capture = cv2.VideoCapture(0)
        video_capture = cv2.VideoCapture("tests/data/%d.png")

    try:
        yield video_capture
    finally:
        video_capture.release()


class CameraThread(threading.Thread):
    def __init__(self, config, results_queue: Queue):
        super(CameraThread, self).__init__()
        self.results_queue = results_queue
        self.stoprequest = threading.Event()

    def run(self):
        with open_stream() as stream:
            counter = 0
            while not self.stoprequest.isSet():
                has_image, image = stream.read()
                if has_image:
                    self.results_queue.put(CameraImage(counter, image))
                    counter = (counter + 1) % 10000

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
