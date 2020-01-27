from contextlib import contextmanager

import cv2
import numpy as np

from game.config import CONFIG
from game.peripherals.display import WHITE, FONT
from game.peripherals.display.face import annotate_face
from game.peripherals.display.registration import add_registration_info
from game.monitoring import monitor_runtime


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


###################################################################################
#                              FRAME HANDLING                                     #
# Working on frame that is extended to all sides by a BUFFER                      #
# -> don't need to check if thumbnail/stars are partly outside the original       #
#    frame, simply draw on extended frame and cut off BUFFER as last step         #
###################################################################################


# needs to be larger than the size of thumbnails/stars
BUFFER = 200


def extend_frame(frame):
    height, width, _ = frame.shape
    extended_frame = np.zeros((height + BUFFER*2, width+BUFFER*2, 4), np.uint8)
    extended_frame[BUFFER: height+BUFFER, BUFFER:width+BUFFER, :] = frame
    return extended_frame


def translate_location(x, y):
    return x + BUFFER, y + BUFFER


def reverse_frame_extension(frame):
    return frame[BUFFER:-BUFFER, BUFFER:-BUFFER, :]


###################################################################################
#                              MAIN DISPLAY ROUTINE                               #
###################################################################################


def write_frame_id(frame, frame_id):
    x, y = 20, 30
    x, y = translate_location(x, y)

    cv2.putText(frame, str(frame_id), x, y, FONT, 1.0, WHITE, lineType=cv2.LINE_AA)


@monitor_runtime
def show_frame(display, image, recognition_result, registration_info):
    frame = image.data
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    if recognition_result or registration_info:
        frame = extend_frame(frame)

        if recognition_result is not None:
            for face in recognition_result.faces:
                annotate_face(frame, face, translate_location)

        if registration_info is not None:
            add_registration_info(frame, registration_info, translate_location)

        if CONFIG["display"]["debug"]:
            cv2.putText(frame, str(image.id), (20 + BUFFER, 30 + BUFFER), FONT, 1.0, WHITE, lineType=cv2.LINE_AA)

        frame = reverse_frame_extension(frame)

    cv2.imshow(display, frame)

