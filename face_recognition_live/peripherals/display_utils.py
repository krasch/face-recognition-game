import cv2
import numpy as np

STARS = {0: cv2.imread("images/stars_0.png", cv2.IMREAD_UNCHANGED),
         1: cv2.imread("images/stars_1.png", cv2.IMREAD_UNCHANGED),
         2: cv2.imread("images/stars_2.png", cv2.IMREAD_UNCHANGED),
         3: cv2.imread("images/stars_3.png", cv2.IMREAD_UNCHANGED)}


def make_round_mask(diameter):
    radius = int(diameter / 2.0)
    center_x = radius
    center_y = radius

    mask = np.zeros((diameter, diameter, 1), np.uint8)
    mask.fill(255)
    cv2.circle(mask, (center_x, center_y), radius, 3, thickness=-1)
    return mask / 255.0


def copy_object_to_location(frame, object_to_copy, x, y, mask):
    height, width, _ = object_to_copy.shape
    original = frame[y: y + height, x: x + width, 0:3] * mask
    new = object_to_copy[:, :, 0:3] * (1-mask)
    frame[y: y + height, x: x + width, 0:3] = original + new


def draw_stars(frame, x, y, num_stars):
    stars = STARS[num_stars]
    height, width, _ = stars.shape

    alpha = stars[:, :, 3:4].astype(float) / 255
    copy_object_to_location(frame, stars, x, y, 1-alpha)


