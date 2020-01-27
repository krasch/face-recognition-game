import cv2

from game.config import CONFIG
from game.events.results import RegistrationResult, UnregistrationResult
from game.peripherals.display import WHITE, FONT
from game.peripherals.display.thumbnail import draw_thumbnail


def calculate_base_location(frame):
    height, width, _ = frame.shape

    x = 10
    y = height - 400

    return x, y


def calculate_thumbnail_location(base_x, base_y, match_index):
    thumbnail_size = CONFIG["display"]["thumbnail_size"]
    thumbnail_dist = 0.25 * thumbnail_size
    box_thumbnail_dist = 10

    x = base_x + int((thumbnail_size + thumbnail_dist) * match_index)
    y = base_y - thumbnail_size - box_thumbnail_dist
    return x, y


def calculate_text_location(base_x, base_y):
    thumbnail_size = CONFIG["display"]["thumbnail_size"]

    x = base_x
    y = base_y - thumbnail_size - 20
    return x, y


def get_text(registration_result):

    if isinstance(registration_result, RegistrationResult):
        if len(registration_result.persons) == 0:
            return "Nobody added :-("
        else:
            return "Added"

    elif isinstance(registration_result, UnregistrationResult):
        if len(registration_result.persons) == 0:
            return "Nobody removed :-("
        else:
            return "Removed:"

    else:
        raise NotImplementedError()


def add_registration_info(frame, registration_result, translate_location):
    base_x, base_y = calculate_base_location(frame)
    base_x, base_y = translate_location(base_x, base_y)

    for i, face in enumerate(registration_result.persons):
        thumbnail_x, thumbnail_y = calculate_thumbnail_location(base_x, base_y, i)
        draw_thumbnail(frame, thumbnail_x, thumbnail_y, face)

    text = get_text(registration_result)
    text_x, text_y = calculate_text_location(base_x, base_y)
    cv2.putText(frame, text, (text_x, text_y), FONT, 0.5, WHITE, lineType=cv2.LINE_AA)
