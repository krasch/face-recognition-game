import queue
from datetime import datetime

import cv2

from face_recognition_live.peripherals.camera import init_camera
from face_recognition_live.peripherals.display import init_display, show_frame
from face_recognition_live.recognition.recognition import init_recognition
from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *
from face_recognition_live.queue import MonitoredQueue
from face_recognition_live.config import CONFIG


def read_camera_until_quit(camera):
    frame_counter = 0
    counter_start_time = datetime.now()

    for image in camera:
        yield image

        pressed_key = cv2.waitKey(1)

        if pressed_key and pressed_key == ord('q'):
            break

        if pressed_key and pressed_key == ord('u'):
            CONFIG.reload()

        frame_counter += 1
        if frame_counter == 100:
            if counter_start_time is not None:
                counter_end_time = datetime.now()
                delta = counter_end_time - counter_start_time
                delta = delta.seconds + delta.microseconds / 1000.0 / 1000.0
                framerate = frame_counter / float(delta)
                print("{} fps".format(framerate))

            counter_start_time = datetime.now()
            frame_counter = 0

def run():
    tasks = MonitoredQueue("tasks")
    results = MonitoredQueue("results")

    faces = None
    currently_recognizing = False

    with init_display() as display, init_camera() as camera, init_recognition(tasks, results):
        for image in read_camera_until_quit(camera):

            try:
                faces = results.get(block=False)
                currently_recognizing = False
            except queue.Empty:
                pass

            if image.id % CONFIG["recognition"]["database"]["backup_frequency"] == 0:
                tasks.put(BackupFaceDatabase())

            if not currently_recognizing:
                currently_recognizing = True
                tasks.put(DetectFaces(image))

            show_frame(display, image, faces)


run()
