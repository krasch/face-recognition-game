from multiprocessing import Queue
import logging.config

import cv2

from face_recognition_live.peripherals.camera import init_camera, monitor_framerate
from face_recognition_live.peripherals.display import init_display, show_frame
from face_recognition_live.recognition.recognition import init_recognition
from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *
from face_recognition_live.events.commands import *
from face_recognition_live.config import CONFIG


def get_next_event(camera, results, errors):
    for image in monitor_framerate(camera):

        # recognition thread had an error, stop all processing
        if not errors.empty():
            break

        # results from face recognition thread came back
        if not results.empty():
            yield results.get(block=False)

        # react to user input
        pressed_key = cv2.waitKey(1)

        if pressed_key and pressed_key == ord('q'):
            break

        if pressed_key and pressed_key == ord('c'):
            CONFIG.reload()

        if pressed_key and pressed_key == ord('r'):
            yield RegisterFacesPressed()

        if pressed_key and pressed_key == ord('u'):
            yield UnregisterFacesPressed()

        # finally actually return the camera image
        yield image


def run():
    config = CONFIG["display"]

    tasks = Queue()
    results = Queue()
    errors = Queue()

    currently_recognizing = False

    recognition_result = None
    registration_result = None

    with init_display() as display, init_camera() as camera, init_recognition(tasks, results, errors):
        for event in get_next_event(camera, results, errors):

            if isinstance(event, RecognitionResult):
                recognition_result = event
                currently_recognizing = False

            elif isinstance(event, RegisterFacesPressed):
                tasks.put(RegisterFaces(recognition_result))

            elif isinstance(event, UnregisterFacesPressed):
                tasks.put(UnregisterMostRecentFaces())

            elif isinstance(event, RegistrationResult) or isinstance(event, UnregistrationResult):
                registration_result = event

            elif isinstance(event, CameraImage):
                image = event

                # should start new recognition?
                if image.id % CONFIG["recognition"]["framerate"] == 0:
                    if not currently_recognizing:
                        currently_recognizing = True
                        image = CameraImage(image.id, image.data.copy())
                        tasks.put(RecognizeFaces(image))

                # clean up older data
                if recognition_result is not None and recognition_result.is_outdated(config["max_age_recognition_result"]):
                    recognition_result = None
                if registration_result is not None and registration_result.is_outdated(config["display_time_registration_result"]):
                    registration_result = None

                # display the new image
                show_frame(display, image, recognition_result, registration_result)

            else:
                raise NotImplementedError()

#logging.config.fileConfig('logging.conf')
FORMAT = '%(asctime)s.%(msecs)03d %(name)s %(message)s'
logging.basicConfig(level="DEBUG", format=FORMAT, datefmt='%H:%M:%S')
run()
