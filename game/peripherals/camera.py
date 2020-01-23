from contextlib import contextmanager
import time
from datetime import datetime
import logging

import cv2
import numpy as np

from game.events.results import CameraImage
from game.config import CONFIG

logger = logging.getLogger("framerate")


def init_jetson_video_capture():
    capture_width = CONFIG["source_settings"]["camera_jetson"]["width"]
    capture_height = CONFIG["source_settings"]["camera_jetson"]["height"]
    flip = CONFIG["source_settings"]["camera_jetson"]["flip"]
    zoom = CONFIG["source_settings"]["camera_jetson"]["zoom"]
    framerate = CONFIG["source_settings"]["camera_jetson"]["framerate"]

    display_width = CONFIG["display"]["width"]
    display_height = CONFIG["display"]["height"]

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
            #f'nvvidconv flip-method={flip} ! ' +
            f'nvvidconv flip-method={flip} left={crop_left} right={crop_right} top={crop_top} bottom={crop_bottom} ! ' +
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

    return SlowerStream(cv2.VideoCapture(CONFIG["source_settings"]["prerecorded"]["location"]), 25)


def monitor_framerate(camera):
    frame_counter = 0
    counter_start_time = datetime.now()

    for image in camera:
        yield image

        frame_counter += 1
        if frame_counter == 100:
            if counter_start_time is not None:
                counter_end_time = datetime.now()
                delta = counter_end_time - counter_start_time
                delta = delta.seconds + delta.microseconds / 1000.0 / 1000.0
                framerate = frame_counter / float(delta)
                logger.info("{} fps".format(framerate))

            counter_start_time = datetime.now()
            frame_counter = 0


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
                if CONFIG["source_settings"]["mirror"]:
                    image = cv2.flip(image, 1)
                yield CameraImage(counter, np.array(image.data).copy())
                counter = (counter + 1) % 10000

    with open_stream() as stream:
        yield read_from_camera(stream)

