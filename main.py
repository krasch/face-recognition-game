import queue

import cv2


from face_recognition_live.peripherals.camera import init_camera
from face_recognition_live.peripherals.display import init_display, show_frame
from face_recognition_live.recognition.recognition import init_recognition
from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *
from face_recognition_live.queue import MonitoredQueue
from face_recognition_live.config import CONFIG


def read_queue_until_quit(q):
    while True:
        pressed_key = cv2.waitKey(1)

        if pressed_key and pressed_key == ord('q'):
            break

        if pressed_key and pressed_key == ord('u'):
            CONFIG.reload()

        try:
            yield q.get(False)
        except queue.Empty:
            yield None


def run():
    tasks = MonitoredQueue("tasks")
    results = MonitoredQueue("results")

    # most recent image and most recent recognized faces
    image = None
    faces = None

    currently_recognizing = True

    with init_display() as display, init_camera(results), init_recognition(tasks, results):
        # initialization: wait for the very first image from the camera
        # will not do face recognition on this one to simplify code, next image will come soon anyway
        for result in read_queue_until_quit(results):
            if isinstance(result, CameraImage):
                image = result
                break

        # main loop
        for i, result in enumerate(read_queue_until_quit(results)):

            if i% 60 == 0:
               print(i)
            #if result.is_outdated(image.id, CONFIG["recognition"]["results_max_age"]):
            #    continue

            if isinstance(result, CameraImage):
                image = result

                if image.id % CONFIG["recognition"]["database"]["backup_frequency"] == 0:
                    tasks.put(BackupFaceDatabase())

                if not currently_recognizing:
                    currently_recognizing = True
                    tasks.put(DetectFaces(image))

            elif isinstance(result, DetectionResult):
                tasks.put(CropFaces(result.image, result.faces))

            elif isinstance(result, CroppingResult):
                tasks.put(ExtractFeatures(result.image, result.faces))

            elif isinstance(result, FeatureExtractionResult):
                tasks.put(RecognizePeople(result.image, result.faces))

            elif isinstance(result, RecognitionResult):
                faces = result
                currently_recognizing = False


            #if faces and faces.is_outdated(image.id, CONFIG["recognition"]["results_max_age"]):
            #    faces = None

            show_frame(display, image, faces)


if __name__ == "__main__":
    run()

