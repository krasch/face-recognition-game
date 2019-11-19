from contextlib import contextmanager

import cv2
import numpy as np

from face_recognition_live.config import CONFIG
from face_recognition_live.recognition.models.matching import MatchQuality
from face_recognition_live.recognition.models.landmarks import DLIB68_FACE_LOCATIONS
from face_recognition_live.events.results import RegistrationResult, UnregistrationResult
from face_recognition_live.peripherals.display_utils import draw_stars, copy_object_to_location, make_round_mask
from face_recognition_live.monitoring import monitor_runtime

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


def add_thumbnail(frame, x, y, thumbnail, thumbnail_size):
    radius = int(thumbnail_size / 2.0)
    y = y - thumbnail_size - DIST_BOX_THUMBNAIL

    thumbnail = cv2.resize(thumbnail, (thumbnail_size, thumbnail_size))
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGRA)

    mask = make_round_mask(thumbnail_size)
    copy_object_to_location(frame, thumbnail, x, y, mask)

    cv2.circle(frame, (x + radius, y + radius), radius, WHITE, thickness=1)


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

        if face.matches:
            matches = filter_matches(face.matches)
            thumbnail_locations = calculate_thumbnail_locations(box.left(), box.top(), len(matches), thumbnail_size)

            for match, (x, y) in zip(matches, thumbnail_locations):
                add_thumbnail(frame, x, y, match.face.thumbnail, thumbnail_size)
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
        text_positive = "Neu registriert"
        text_negative = "Niemand registiert :-("
    else:
        text_positive = "Entfernt:"
        text_negative = "Niemand entfernt :-("

    if len(registration_result.thumbnails) == 0:
        cv2.putText(frame, text_negative, (x + BUFFER, y_text), FONT, 0.5, WHITE, lineType=cv2.LINE_AA)
        return

    cv2.putText(frame, text_positive, (x + BUFFER, y_text), FONT, 0.5, WHITE, lineType=cv2.LINE_AA)
    thumbnail_locations = calculate_thumbnail_locations(x, y_thumbnail_top, len(registration_result.thumbnails), thumbnail_size)
    for face, (x, y) in zip(registration_result.thumbnails, thumbnail_locations):
        add_thumbnail(frame, x, y, face, thumbnail_size)


def store_data(image, recognition_result):
    import pickle
    if image.id == 85:
        with open("tests/data/display_test/data.pkl", "wb") as f:
            pickle.dump({"image": image, "recognition_result": recognition_result}, f)


def extend_frame(frame):
    height, width, _ = frame.shape
    extended_frame = np.zeros((height + BUFFER*2, width+BUFFER*2, 4), np.uint8)
    extended_frame[BUFFER: height+BUFFER, BUFFER:width+BUFFER, :] = frame
    return extended_frame


def reverse_frame_extension(frame):
    return frame[BUFFER:-BUFFER, BUFFER:-BUFFER, :]


#from profilehooks import profile
#@profile
def show_frame(display, image, recognition_result, registration_info):
    frame = cv2.cvtColor(image.data, cv2.COLOR_RGB2BGRA)
    
    if recognition_result:
        cv2.imshow(display, frame)

    frame = extend_frame(frame)
    if recognition_result is not None:
        add_recognition_result(frame, recognition_result)
    if registration_info is not None:
        add_registration_info(frame, registration_info)

    if CONFIG["display"]["debug"]:
        cv2.putText(frame, str(image.id), (20 + BUFFER, 30 + BUFFER), FONT, 1.0, WHITE, lineType=cv2.LINE_AA)

    frame = reverse_frame_extension(frame)
    cv2.imshow(display, frame)

