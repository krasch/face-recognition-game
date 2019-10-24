from contextlib import contextmanager

import cv2

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


def show_frame(display, image, faces):
    if image is None:
        return

    frame = image.data.copy()

    if faces:
        print("hallo")
        for face in faces.faces:
            print(face)
            print(face.match)
            if face.match:
                color = GREEN
                frame[30:180, 0:0 + 150] = face.match.image
            else:
                color = RED

            box = face.bounding_box
            cv2.rectangle(frame, (box.left(), box.top()), (box.right(), box.bottom()), color, 2)

    cv2.imshow(display, frame)
