import queue

import cv2
import yaml
import quickargs


from face_recognition_live.peripherals.camera import init_camera
from face_recognition_live.peripherals.display import init_display, show_frame
from face_recognition_live.recognition import init_recognition
from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *


def read_queue_until_quit(q):
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            yield q.get(True, 1.0/50.0)
        except queue.Empty:
            continue


def run(config):
    RESULTS_MAX_AGE = config["recognition"]["results_max_age"]
    DATABASE_BACKUP_FREQUENCY = config["recognition"]["database"]["backup_frequency"]
    RECOGNITION_FREQUENCY = config["recognition"]["framerate"]

    tasks = queue.Queue()
    results = queue.Queue()

    # most recent image and most recent recognized faces
    image = None
    faces = None

    with init_display(config) as display, init_camera(config, results), init_recognition(config, tasks, results):
        # initialization: wait for the very first image from the camera
        # will not do face recognition on this one to simplify code, next image will come soon anyway
        for result in read_queue_until_quit(results):
            if isinstance(result, CameraImage):
                image = result
                break

        # main loop
        for result in read_queue_until_quit(results):

            if result.is_outdated(image.id, RESULTS_MAX_AGE):
                continue

            if isinstance(result, CameraImage):
                image = result

                if image.id % DATABASE_BACKUP_FREQUENCY == 0:
                    tasks.put(BackupFaceDatabase())

                if image.id % RECOGNITION_FREQUENCY == 0:
                    tasks.put(DetectFaces(image))

            elif isinstance(result, DetectionResult):
                tasks.put(CropFaces(result.image, result.faces))

            elif isinstance(result, CroppingResult):
                tasks.put(ExtractFeatures(result.image, result.faces))

            elif isinstance(result, FeatureExtractionResult):
                tasks.put(RecognizePeople(result.image, result.faces))

            elif isinstance(result, RecognitionResult):
                faces = result

            else:
                raise NotImplementedError()

            if faces and faces.is_outdated(image.id, RESULTS_MAX_AGE):
                faces = None

            show_frame(display, image, faces)


if __name__ == "__main__":
    with open("config.yaml") as f:
        configuration = yaml.load(f, Loader=quickargs.YAMLArgsLoader)
    run(configuration)

