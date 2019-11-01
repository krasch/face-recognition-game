from contextlib import contextmanager

import cv2

from face_recognition_live.recognition.models.landmarks import DLIB68_FACE_LOCATIONS

# BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)


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


def add_bounding_box(frame, face):
    if face.match:
        color = GREEN
    else:
        color = RED

    box = face.bounding_box
    cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), color, 2)


def add_landmarks(frame, face):
    for landmark in DLIB68_FACE_LOCATIONS:
        location = face.landmarks.parts()[landmark.value].x, face.landmarks.parts()[landmark.value].y
        cv2.circle(frame, location, 3, (255, 255, 255), 0)


def add_match(frame, face):
    if face.match:
        frame[30:180, 0:0 + 150] = face.match.image


def add_is_frontal_debug(frame, face):
    left, right, top, bottom = face.frontal.frontal_box

    if face.frontal.is_frontal:
        color = GREEN
    else:
        color = RED

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)


def show_frame(display, image, recognition_result):
    if image is None:
        return

    frame = image.data.copy()

    if recognition_result:
        for face in recognition_result.faces:
            add_bounding_box(frame, face)

            # debug mode
            #add_landmarks(frame, face)
            add_match(frame, face)
            #add_is_frontal_debug(frame, face)

            # cv2.putText(frame, str(arrow["nose"][0] < arrow["mouth_right"][0]), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    cv2.imshow(display, frame)
