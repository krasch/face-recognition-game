from contextlib import contextmanager
import queue

import cv2
import yaml
import quickargs


from recognition.camera import init_camera
from recognition.recognition import init_recognition
from recognition.tasks import FaceDetection, FaceRecognition
from recognition.results import CameraImage, DetectedFaces


@contextmanager
def init_display(config):
    name = "window"
    if config["display"]["fullscreen"]:
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow("window")
   
    try:
        yield name
    finally:
        cv2.destroyWindow(name)


def read_queue_until_quit(q):
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            yield q.get(True, 1.0/15.0)
        except queue.Empty:
            continue


def show_frame(display, image, faces):
    if image is None:
        return

    frame = image.data.copy()

    if faces:
        for face in faces.face_locations:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

    cv2.imshow(display, frame)


def run(config):
    TTL = 4

    tasks = queue.Queue()
    results = queue.Queue()

    image = None
    faces = None
    persons = None

    with init_display(config) as display, init_camera(config, results), init_recognition(config, tasks, results):
        # initialization: wait for the very first image from the camera
        for result in read_queue_until_quit(results):
            if isinstance(result, CameraImage):
                image = result
                break

        # main loop
        for result in read_queue_until_quit(results):
            if result.is_outdated(image.id, TTL):
                continue

            if isinstance(result, CameraImage):
                image = result
                tasks.put(FaceDetection(image))
                import time
                time.sleep(1)

            elif isinstance(result, DetectedFaces):
                faces = result
                tasks.put(FaceRecognition(result.original_image, result.face_locations))

            else:
                raise NotImplementedError()

            if faces and faces.is_outdated(image.id, TTL):
                faces = None
            if persons and persons.is_outdated(image.id, TTL):
                persons = None

            show_frame(display, image, faces)



if __name__ == "__main__":
    with open("config.yaml") as f:
        configuration = yaml.load(f, Loader=quickargs.YAMLArgsLoader)
    run(configuration)

