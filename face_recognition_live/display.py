from contextlib import contextmanager

import cv2


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


def show_frame(display, image, faces):
    if image is None:
        return

    frame = image.data.copy()

    if faces:
        for face in faces.persons:
            box = face.bounding_box
            cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), (0, 0, 255), 2)

    cv2.imshow(display, frame)
