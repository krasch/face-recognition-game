from multiprocessing import Queue

import cv2

from game.peripherals.camera import init_camera, monitor_framerate
from game.peripherals.display.display import init_display, show_frame
from game.recognition.recognition import init_recognition
from game.events.tasks import *
from game.events.results import *
from game.events.commands import *
from game.config import CONFIG


CLICKER_LEFT_KEY = 85
CLICKER_RIGHT_KEY = 86


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

        if pressed_key and pressed_key == ord('d'):
            CONFIG.toggle_debug()

        if pressed_key and pressed_key == ord('r') or pressed_key == CLICKER_RIGHT_KEY:
            yield RegisterFacesPressed()

        if pressed_key and pressed_key == ord('u') or pressed_key == CLICKER_LEFT_KEY:
            yield UnregisterFacesPressed()

        # finally actually return the camera image
        yield image


def game_loop():
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
                if not currently_recognizing:
                    currently_recognizing = True
                    image = CameraImage(image.id, image.data.copy())
                    tasks.put(RecognizeFaces(image))

                if image.id % CONFIG["recognition"]["database"]["backup_frequency"] == 0:
                    tasks.put(BackupFaceDatabase())

                # clean up older data
                if recognition_result is not None and recognition_result.is_outdated(config["max_age_recognition_result"]):
                    recognition_result = None
                if registration_result is not None and registration_result.is_outdated(config["display_time_registration_result"]):
                    registration_result = None

                # display the new image
                show_frame(display, image, recognition_result, registration_result)

            else:
                raise NotImplementedError()
