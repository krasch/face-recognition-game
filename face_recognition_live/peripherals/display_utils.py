import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


THREE_STARS = u"★★★"
TWO_STARS = u"★★☆"
ONE_STAR = u"★☆☆"
NO_STARS = u"☆☆☆"

STARS = {0: NO_STARS, 1: ONE_STAR, 2: TWO_STARS, 3: THREE_STARS}

font = ImageFont.truetype("face_recognition_live/peripherals/OpenSansEmoji.ttf", 40)


def make_round_mask(image_height, image_width, center_x, center_y, radius):
    mask = np.zeros((image_height, image_width, 3), np.uint8)
    mask.fill(255)
    cv2.circle(mask, (center_x + radius, center_y + radius), radius, 3, thickness=-1)
    return mask < 255


def copy_object_to_location(image_height, image_width, x, y, object_to_copy):
    object_height, object_width, _ = object_to_copy.shape
    object_height = min(object_height, image_height-y)
    object_width = min(object_width, image_width-x)

    white_image_with_object = np.zeros((image_height, image_width, 3), np.uint8)
    white_image_with_object[y: y + object_height, x: x + object_width] = object_to_copy[:object_height, :object_width, :]
    return white_image_with_object


def draw_stars(frame, x, y, num_stars):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.text((x, y), STARS[num_stars], font=font)
    return np.array(image)
