from contextlib import contextmanager

import cv2
import numpy as np

from game.config import CONFIG
from game.recognition.models.matching import MatchQuality
from game.recognition.models.landmarks import DLIB68_FACE_LOCATIONS
from game.events.results import RegistrationResult, UnregistrationResult
from game.peripherals.display_utils import *
from game.monitoring import monitor_runtime

# BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (125, 125, 125)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

LANDMARKS = [DLIB68_FACE_LOCATIONS.NOSE_TIP,
             DLIB68_FACE_LOCATIONS.LEFT_EYE_LEFT_CORNER, DLIB68_FACE_LOCATIONS.RIGHT_EYE_RIGHT_CORNER,
             DLIB68_FACE_LOCATIONS.MOUTH_LEFT, DLIB68_FACE_LOCATIONS.MOUTH_RIGHT]

BUFFER = 200
DIST_BOX_THUMBNAIL = 10
DIST_THUMBNAIL_STARS = 5


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


def add_bounding_box(frame, box):
    cv2.rectangle(frame, (box.left() + BUFFER, box.top()+BUFFER), (box.right()+BUFFER, box.bottom()+BUFFER), WHITE, 2)


def add_landmarks(frame, face):
    for landmark in LANDMARKS:
        location = face.landmarks.parts()[landmark.value].x+BUFFER, face.landmarks.parts()[landmark.value].y+BUFFER
        cv2.circle(frame, location, 2, WHITE, -1)


def calculate_thumbnail_locations(leftmost, top, num_thumbnails, thumbnail_size):
    dist_x = int(0.25 * thumbnail_size)

    def calculate_x(i):
        return leftmost + (thumbnail_size + dist_x) * i + BUFFER

    y = top + BUFFER
    return [(calculate_x(i), y) for i in range(num_thumbnails)]


def add_thumbnail(frame, x, y, person, thumbnail_size):
    y = y - thumbnail_size - DIST_BOX_THUMBNAIL

    thumbnail = get_thumbnail(person, thumbnail_size)
    mask = make_round_mask(thumbnail_size)
    copy_object_to_location(frame, thumbnail, x, y, mask)


def draw_stars(frame, x, y, num_stars, stars_height):
    stars = get_stars(num_stars, stars_height)
    alpha = stars[:, :, 3:4].astype(float) / 255
    copy_object_to_location(frame, stars, x, y, 1-alpha)


def add_match_stars(frame, x, y, match_quality, stars_height):
    if match_quality == MatchQuality.EXCELLENT:
        return draw_stars(frame, x, y, 3, stars_height)
    elif match_quality == MatchQuality.GOOD:
        return draw_stars(frame, x, y, 2, stars_height)
    elif match_quality == MatchQuality.POOR:
        return draw_stars(frame, x, y, 1, stars_height)
    else:
        return draw_stars(frame, x, y, 0, stars_height)


def add_match_distance(frame, x, y, match_distance):
    match_distance = np.round(match_distance, 2)
    cv2.putText(frame, str(match_distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, lineType=cv2.LINE_AA)


def add_is_frontal_debug(frame, face):
    left, right, top, bottom = face.frontal.frontal_box

    if face.frontal.is_frontal:
        color = GREEN
    else:
        color = RED

    cv2.rectangle(frame, (left + BUFFER, top + BUFFER), (right + BUFFER, bottom + BUFFER), color, 2)


def filter_matches(matches):
    if CONFIG["display"]["debug"]:
        matches = matches[:CONFIG["display"]["num_matches_debug"]]
    else:
        matches = [m for m in matches if m.quality != MatchQuality.NO_MATCH]
        matches = matches[:CONFIG["display"]["num_matches"]]
    return matches


def add_recognition_result(frame, recognition_result):
    thumbnail_size = CONFIG["display"]["thumbnail_size"]
    stars_height = CONFIG["display"]["stars_height"]
    debug = CONFIG["display"]["debug"]

    for face in recognition_result.faces:
        box = face.bounding_box
        add_bounding_box(frame, box)
        if debug:
            add_landmarks(frame, face)

        if face.matches:
            matches = filter_matches(face.matches)
            thumbnail_locations = calculate_thumbnail_locations(box.left(), box.top(), len(matches), thumbnail_size)

            for match, (x, y) in zip(matches, thumbnail_locations):
                add_thumbnail(frame, x, y, match.face, thumbnail_size)
                y_stars = y - thumbnail_size - DIST_BOX_THUMBNAIL - stars_height
                add_match_stars(frame, x, y_stars, match.quality, stars_height)

                if debug:
                    y_distance = y - thumbnail_size - DIST_BOX_THUMBNAIL - stars_height - DIST_THUMBNAIL_STARS
                    add_match_distance(frame, x, y_distance, match.distance)


def add_registration_info(frame, registration_result):
    thumbnail_size = CONFIG["display"]["thumbnail_size"]
    height, width, _ = frame.shape

    x = 10
    y_thumbnail_top = height - BUFFER * 2
    y_text = height - BUFFER - thumbnail_size - DIST_BOX_THUMBNAIL * 2  # seems arbitrary

    if isinstance(registration_result, RegistrationResult):
        text_positive = "Added"
        text_negative = "Nobody added :-("
    else:
        text_positive = "Removed:"
        text_negative = "Nobody removed :-("

    if len(registration_result.persons) == 0:
        cv2.putText(frame, text_negative, (x + BUFFER, y_text), FONT, 0.5, WHITE, lineType=cv2.LINE_AA)
        return

    cv2.putText(frame, text_positive, (x + BUFFER, y_text), FONT, 0.5, WHITE, lineType=cv2.LINE_AA)
    thumbnail_locations = calculate_thumbnail_locations(x, y_thumbnail_top, len(registration_result.persons), thumbnail_size)
    for face, (x, y) in zip(registration_result.persons, thumbnail_locations):
        add_thumbnail(frame, x, y, face, thumbnail_size)


def extend_frame(frame):
    height, width, _ = frame.shape
    extended_frame = np.zeros((height + BUFFER*2, width+BUFFER*2, 4), np.uint8)
    extended_frame[BUFFER: height+BUFFER, BUFFER:width+BUFFER, :] = frame
    return extended_frame


def reverse_frame_extension(frame):
    return frame[BUFFER:-BUFFER, BUFFER:-BUFFER, :]


@monitor_runtime
def show_frame(display, image, recognition_result, registration_info):
    frame = image.data
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    if recognition_result or registration_info:
        frame = extend_frame(frame)
        if recognition_result is not None:
            add_recognition_result(frame, recognition_result)
        if registration_info is not None:
            add_registration_info(frame, registration_info)

        if CONFIG["display"]["debug"]:
            cv2.putText(frame, str(image.id), (20 + BUFFER, 30 + BUFFER), FONT, 1.0, WHITE, lineType=cv2.LINE_AA)

        frame = reverse_frame_extension(frame)

    cv2.imshow(display, frame)

