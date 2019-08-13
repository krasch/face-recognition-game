from contextlib import contextmanager
import queue

import cv2
import yaml
import quickargs


from recognition.camera import init_camera
from recognition.recognition import init_recognition
from recognition.types import ResultTypes, Result, TaskTypes, Task


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
"""
#@profile
def main():
    detect_faces = init_face_detector_cnn()
    crop_faces = init_face_cropper()
    extract_features = init_feature_extractor()
    match_faces = init_face_matching()

    time.sleep(3)  # to get model loading to finish

    with init_display(FULLSIZE) as display_name:
        with open_stream() as stream:
            for i, frame in enumerate(read_stream_until_quit(stream)):  # todo can enumerate overflow?
                cv2.imshow(display_name, frame)
                print(frame.shape)

                face_locations = detect_faces(frame)
                if not face_locations:
                    continue

                for face in face_locations:
                    cropped = crop_faces(frame, face)  # openface has its own alignment, seems to use 96 points
                    features = extract_features(cropped)[0]
                    matched = match_faces(features)
                    if matched:
                       color = (0, 0, 255)
                    else:
                       color = (255, 0, 0)
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

                cv2.imshow(display_name, frame)
"""


def show_frame(display, frame, face_locations):
    if frame is None:
        return

    frame = frame.copy()

    if face_locations:
        for face in face_locations:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

    cv2.imshow(display, frame)


def run(config):
    tasks = queue.Queue()
    results = queue.Queue()

    frame = None
    faces = None

    with init_display(config) as display, init_camera(config, results), init_recognition(config, tasks, results):
        for result in read_queue_until_quit(results):

            if result.type == ResultTypes.CAMERA_IMAGE:
                frame = result.data
                tasks.put(Task(TaskTypes.DETECT_FACES, result.data))

            elif result.type == ResultTypes.FACE_DETECTIONS:
                original_frame, faces = result.data
                for face in faces:
                    tasks.put(Task(TaskTypes.RECOGNIZE_PERSON, (original_frame, face)))

            else:
                raise NotImplementedError()

            show_frame(display, frame, faces)


if __name__ == "__main__":
    with open("config.yaml") as f:
        configuration = yaml.load(f, Loader=quickargs.YAMLArgsLoader)
    run(configuration)

