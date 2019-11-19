import cv2
import numpy as np


# only prepare mask once for every diameter
def make_round_mask(diameter):
    masks = {}

    def make_mask():
        radius = int(diameter / 2.0)
        center_x = radius
        center_y = radius

        mask = np.zeros((diameter, diameter, 1), np.uint8)
        mask.fill(255)
        cv2.circle(mask, (center_x, center_y), radius, 3, thickness=-1)
        return mask / 255.0

    if diameter not in masks:
        masks[diameter] = make_mask()

    return masks[diameter]


# only prepare stars once for every combination of num_stars and stars_height
def get_star(num_stars, stars_height):
    stars_original = {0: cv2.imread("images/stars_0.png", cv2.IMREAD_UNCHANGED),
                      1: cv2.imread("images/stars_1.png", cv2.IMREAD_UNCHANGED),
                      2: cv2.imread("images/stars_2.png", cv2.IMREAD_UNCHANGED),
                      3: cv2.imread("images/stars_3.png", cv2.IMREAD_UNCHANGED)}
    stars_resized = {0: {}, 1: {}, 2: {}, 3: {}}

    if stars_height not in stars_resized[num_stars]:
        s = cv2.resize(stars_original[num_stars], (stars_height*3, stars_height))
        stars_resized[num_stars][stars_height] = s

    return stars_resized[num_stars][stars_height]


def copy_object_to_location(frame, object_to_copy, x, y, mask):
    frame_height, frame_width, _ = frame.shape
    object_height, object_width, _ = object_to_copy.shape

    if x < 0 or x + object_width > frame_width or y < 0 or y + object_height > frame_height:
        return

    original = frame[y: y + object_height, x: x + object_width, 0:3] * mask
    new = object_to_copy[:, :, 0:3] * (1-mask)
    frame[y: y + object_height, x: x + object_width, 0:3] = original + new


def draw_stars(frame, x, y, num_stars, stars_height):
    stars = get_star(num_stars, stars_height)
    alpha = stars[:, :, 3:4].astype(float) / 255
    copy_object_to_location(frame, stars, x, y, 1-alpha)


