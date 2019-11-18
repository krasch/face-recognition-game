from contextlib import contextmanager

import cv2
import numpy as np
import sdl2
import sdl2.ext

from face_recognition_live.recognition.models.landmarks import DLIB68_FACE_LOCATIONS
from face_recognition_live.recognition.models.matching import MatchQuality
from face_recognition_live.config import CONFIG
from face_recognition_live.peripherals.display_utils import draw_stars, copy_object_to_location, make_round_mask
from face_recognition_live.monitoring import monitor_runtime

# BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (125, 125, 125)
WHITE = (255, 255, 255)

LANDMARKS = [DLIB68_FACE_LOCATIONS.NOSE_TIP,
             DLIB68_FACE_LOCATIONS.LEFT_EYE_LEFT_CORNER, DLIB68_FACE_LOCATIONS.RIGHT_EYE_RIGHT_CORNER,
             DLIB68_FACE_LOCATIONS.MOUTH_LEFT, DLIB68_FACE_LOCATIONS.MOUTH_RIGHT]

BUFFER = 200


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
    cv2.rectangle(frame, (box.left() + BUFFER, box.top()+BUFFER), (box.right()+BUFFER, box.bottom()+BUFFER), WHITE, 2)


def add_landmarks(frame, face):
    for landmark in LANDMARKS:
        location = face.landmarks.parts()[landmark.value].x+BUFFER, face.landmarks.parts()[landmark.value].y+BUFFER
        cv2.circle(frame, location, 2, WHITE, -1)


def add_matches(frame, face, config):
    height, width, _ = frame.shape
    thumbnail_size = config["thumbnail_size"]
    radius = int(thumbnail_size / 2.0)

    def calculate_thumbnail_center(thumbnail_index):
        x = face.bounding_box.left() + (thumbnail_size + 25) * thumbnail_index
        y = face.bounding_box.top() - int(thumbnail_size * 1.2)
        return x + BUFFER, y + BUFFER

    def add_thumbnail(x, y, thumbnail):
        thumbnail = cv2.resize(thumbnail, (thumbnail_size, thumbnail_size))
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2RGBA)
        mask = make_round_mask(thumbnail_size)
        copy_object_to_location(frame, thumbnail, x, y, mask)
        cv2.circle(frame, (x + radius, y + radius), radius, WHITE, thickness=2)

    def add_match_stars(x, y, match_quality):
        y = y - radius + 5

        if match_quality == MatchQuality.EXCELLENT:
            return draw_stars(frame, x, y, 3)
        elif match_quality == MatchQuality.GOOD:
            return draw_stars(frame, x, y, 2)
        elif match_quality == MatchQuality.POOR:
            return draw_stars(frame, x, y, 1)
        else:
            return draw_stars(frame, x, y, 0)

    def add_match_distance(x, y, match_distance):
        y = y - radius - 5
        match_distance = np.round(match_distance, 2)
        cv2.putText(frame, str(match_distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, lineType=cv2.LINE_AA)

    # in debug mode display many matches, in all qualities
    # in not debug mode only display the best x okayish matches
    if config["debug"]:
        matches = face.matches[:config["num_matches_debug"]]
    else:
        matches = [m for m in face.matches if m.quality != MatchQuality.NO_MATCH]
        matches = matches[:config["num_matches"]]

    for i, match in enumerate(matches):
        center_x, center_y = calculate_thumbnail_center(i)
        add_thumbnail(center_x, center_y, match.face.image)
        add_match_stars(center_x, center_y, match.quality)

        if config["debug"]:
            add_match_distance(center_x, center_y, match.distance)

    return frame


def add_is_frontal_debug(frame, face):
    left, right, top, bottom = face.frontal.frontal_box

    if face.frontal.is_frontal:
        color = GREEN
    else:
        color = RED

    cv2.rectangle(frame, (left + BUFFER, top + BUFFER), (right + BUFFER, bottom + BUFFER), color, 2)


@monitor_runtime
def show_frame(display, image, recognition_result):
    frame = image.data
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

    height, width, _ = frame.shape
    extended_frame = np.zeros((height + BUFFER*2, width+BUFFER*2, 4), np.uint8)
    extended_frame[BUFFER: height+BUFFER, BUFFER:width+BUFFER, :] = frame

    if recognition_result:
        for face in recognition_result.faces:
            add_bounding_box(extended_frame, face)
            if face.matches:
                extended_frame = add_matches(extended_frame, face, CONFIG["display"])

            if CONFIG["display"]["debug"]:
                add_is_frontal_debug(extended_frame, face)
                add_landmarks(extended_frame, face)

    if CONFIG["display"]["debug"]:
        cv2.putText(extended_frame, str(image.id), (20 + BUFFER, 30 + BUFFER),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, lineType=cv2.LINE_AA)

    frame = extended_frame[BUFFER:-BUFFER, BUFFER:-BUFFER, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
    cv2.imshow(display, frame)

