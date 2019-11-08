from contextlib import contextmanager
import pickle

import cv2
import numpy as np

from face_recognition_live.recognition.models.landmarks import DLIB68_FACE_LOCATIONS
from face_recognition_live.config import CONFIG

# BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (125, 125, 125)
WHITE = (255, 255, 255)



@contextmanager
def init_display():
    name = "window"
    if CONFIG["display"]["fullscreen"]:
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow("window")

    try:
        yield name
    finally:
        cv2.destroyWindow(name)


def add_bounding_box(frame, face):
    box = face.bounding_box
    cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), WHITE, 2)


def add_landmarks(frame, face):
    for landmark in DLIB68_FACE_LOCATIONS:
        location = face.landmarks.parts()[landmark.value].x, face.landmarks.parts()[landmark.value].y
        cv2.circle(frame, location, 3, WHITE, 0)


def make_round_mask(image, x, y, radius):
    height, width, _ = image.shape
    mask = np.zeros((height, width, 3), np.uint8)
    mask.fill(255)
    cv2.circle(mask, (x + radius, y + radius), radius, 3, thickness=-1)
    return mask < 255


def add_match(frame, face):
    size = CONFIG["display"]["match_display_size"]

    height, width, _ = frame.shape
    if face.match:
        for i, match in enumerate(face.match.matches):
            x = face.bounding_box.left() + size*i + 5
            y = face.bounding_box.top() - int(size*1.2)
            radius = int(size / 2.0)

            # for square thumbnail
            # thumbnail = cv2.resize(face.match.best_match.image, (MATCH_DISPLAY_SIZE, MATCH_DISPLAY_SIZE))
            # frame[y:y+MATCH_DISPLAY_SIZE, x:x + MATCH_DISPLAY_SIZE] = thumbnail

            # for round thumbnail
            thumbnail = cv2.resize(match.image, (size, size))
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
            thumbnail_at_right_place = np.zeros((height+size*2, width+size*2, 3), np.uint8)
            thumbnail_at_right_place[y+size: y + size*2, x+size: x + size*2] = thumbnail
            thumbnail_at_right_place = thumbnail_at_right_place[size:-size, size:-size,:]
            mask = make_round_mask(frame, x, y, radius)
            np.copyto(src=thumbnail_at_right_place, dst=frame, where=mask)
            cv2.circle(frame, (x+radius, y+radius), radius, WHITE, thickness=2)

    else:
        x = face.bounding_box.left() + 5
        y = face.bounding_box.top() - int(size * 1.2)
        radius = int(size / 2.0)
        cv2.circle(frame, (x + radius, y + radius), radius, WHITE, thickness=-1)
        cv2.putText(frame, "?", (x+radius-20, y+radius+20), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,0),  lineType=cv2.LINE_AA)


def add_is_frontal_debug(frame, face):
    left, right, top, bottom = face.frontal.frontal_box

    if face.frontal.is_frontal:
        color = GREEN
    else:
        color = RED

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)


def add_match_distance(frame, face):
    size = CONFIG["display"]["match_display_size"]

    if face.match:
        for i, match in enumerate(face.match.matches):
            x = face.bounding_box.left() + size*i+10
            y = face.bounding_box.top() - size
            dist = round(sorted(face.match.distances)[i], 2)
            cv2.putText(frame, str(dist), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE,  lineType=cv2.LINE_AA)


def show_frame(display, image, recognition_result):
    if image is None:
        return

    if image.id == 60:
        # todo stores as bgr
        with open("tests/data/display_test/data.pkl", "wb") as f:
            pickle.dump({"image": image, "recognition_result": recognition_result}, f)

    frame = image.data.copy()
    frame.data = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if recognition_result:
        for face in recognition_result.faces:
            add_bounding_box(frame, face)
            add_match(frame, face)
            add_landmarks(frame, face)

            if CONFIG["display"]["debug"]:
                add_is_frontal_debug(frame, face)
                add_match_distance(frame, face)

    if CONFIG["display"]["debug"]:
        cv2.putText(frame, str(image.id), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, lineType=cv2.LINE_AA)

    cv2.imshow(display, frame)
