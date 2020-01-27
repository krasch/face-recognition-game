import cv2
import numpy as np

from game.recognition.models.landmarks import DLIB68_FACE_LOCATIONS
from game.recognition.models.matching import MatchQuality
from game.peripherals.display import WHITE, FONT
from game.peripherals.display.thumbnail import draw_thumbnail
from game.peripherals.display.stars import draw_stars
from game.config import CONFIG


LANDMARKS = [DLIB68_FACE_LOCATIONS.NOSE_TIP,
             DLIB68_FACE_LOCATIONS.LEFT_EYE_LEFT_CORNER, DLIB68_FACE_LOCATIONS.RIGHT_EYE_RIGHT_CORNER,
             DLIB68_FACE_LOCATIONS.MOUTH_LEFT, DLIB68_FACE_LOCATIONS.MOUTH_RIGHT]

###################################################################################
#                              DRAWING                                            #
###################################################################################


def draw_landmarks(frame, landmarks, translate_location):
    for landmark in LANDMARKS:
        x, y = landmarks.parts()[landmark.value].x, landmarks.parts()[landmark.value].y
        x, y = translate_location(x, y)
        cv2.circle(frame, (x, y), 2, WHITE, -1)


def draw_bounding_box(frame, bounding_box, translate_location):
    x1, y1 = bounding_box.left(), bounding_box.top()
    x2, y2 = bounding_box.right(), bounding_box.bottom()

    x1, y1 = translate_location(x1, y1)
    x2, y2 = translate_location(x2, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)


def write_match_distance(frame, x, y, match_distance):
    match_distance = np.round(match_distance, 2)
    cv2.putText(frame, str(match_distance), (x, y), FONT, 0.5, WHITE, lineType=cv2.LINE_AA)


###################################################################################
#                              CALCULATING                                        #
###################################################################################

def filter_matches(matches):
    if CONFIG["display"]["debug"]:
        matches = matches[:CONFIG["display"]["num_matches_debug"]]
    else:
        matches = [m for m in matches if m.quality != MatchQuality.NO_MATCH]
        matches = matches[:CONFIG["display"]["num_matches"]]
    return matches


def calculate_thumbnail_location(base_x, base_y, match_index):
    thumbnail_size = CONFIG["display"]["thumbnail_size"]
    thumbnail_dist = 0.25 * thumbnail_size
    box_thumbnail_dist = 10

    x = base_x + int((thumbnail_size + thumbnail_dist) * match_index)
    y = base_y - thumbnail_size - box_thumbnail_dist
    return x, y


def calculate_stars_location(thumbnail_x, thumbnail_y):
    stars_height = CONFIG["display"]["stars_height"]
    return thumbnail_x, thumbnail_y - stars_height


def calculate_text_location(stars_x, stars_y):
    return stars_x, stars_y - 5


###################################################################################
#                 MAIN ANNOTATION FUNCTION                                        #
###################################################################################

def annotate_face(frame, face, translate_location):
    draw_bounding_box(frame, face.bounding_box, translate_location)

    if face.matches:
        matches = filter_matches(face.matches)

        for i, match in enumerate(matches):
            base_x, base_y = translate_location(face.bounding_box.left(), face.bounding_box.top())

            thumbnail_x, thumbnail_y = calculate_thumbnail_location(base_x, base_y, i)
            draw_thumbnail(frame, thumbnail_x, thumbnail_y, match.face)

            stars_x, stars_y = calculate_stars_location(thumbnail_x, thumbnail_y)
            draw_stars(frame, stars_x, stars_y, match.quality)

            if CONFIG["display"]["debug"]:
                text_x, text_y = calculate_text_location(stars_x, stars_y)
                write_match_distance(frame, text_x, text_y, match.distance)

    if CONFIG["display"]["debug"]:
        draw_landmarks(frame, face.landmarks, translate_location)


