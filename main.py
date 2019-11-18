from multiprocessing import Queue
import logging.config

import cv2

from face_recognition_live.peripherals.camera import init_camera, monitor_framerate
from face_recognition_live.peripherals.display import init_display, show_frame
from face_recognition_live.recognition.recognition import init_recognition
from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *
from face_recognition_live.config import CONFIG


def read_camera_until_quit(camera):
    for image in monitor_framerate(camera):
        yield image

        pressed_key = cv2.waitKey(1)

        if pressed_key and pressed_key == ord('q'):
            break

        if pressed_key and pressed_key == ord('u'):
            CONFIG.reload()


def run():
    tasks = Queue()
    results = Queue()
    errors = Queue()

    faces = None
    currently_recognizing = False

    with init_display() as display, init_camera() as camera, init_recognition(tasks, results, errors):
        for image in read_camera_until_quit(camera):

            # recognition thread had an error, stop all processing
            if not errors.empty():
                break

            # face recognition results came back
            if not results.empty():
                faces = results.get(block=False)
                currently_recognizing = False

            #if image.id % CONFIG["recognition"]["database"]["backup_frequency"] == 0:
            #    tasks.put(BackupFaceDatabase())

            if image.id % CONFIG["recognition"]["framerate"] == 0 and image.id > 0:
                if not currently_recognizing:
                    currently_recognizing = True
                    image = CameraImage(image.id, image.data.copy())
                    tasks.put(RecognizeFaces(image))

            #if faces and faces.is_outdated(image.id, CONFIG["recognition"]["results_max_age"]):
            #    faces = None

            show_frame(display, image, faces)


#logging.config.fileConfig('logging.conf')
FORMAT = '%(asctime)s.%(msecs)03d %(name)s %(message)s'
logging.basicConfig(level="DEBUG", format=FORMAT, datefmt='%H:%M:%S')
run()
