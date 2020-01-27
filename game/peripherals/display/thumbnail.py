import cv2
import numpy as np

from game.config import CONFIG
from game.peripherals.display import WHITE
from game.peripherals.display.util import copy_object_to_location


def make_round_mask(diameter):
    radius = int(diameter / 2.0)
    center_x = radius
    center_y = radius

    mask = np.zeros((diameter, diameter, 1), np.uint8)
    mask.fill(255)
    cv2.circle(mask, (center_x, center_y), radius, 3, thickness=-1)
    return mask / 255.0


# to avoid having to make the same-diameter mask multiple times
def init_masks_cache():
    masks = {}

    def get_mask(diameter):
        if diameter not in masks:
            masks[diameter] = make_round_mask(diameter)
        return masks[diameter]

    return get_mask


def create_thumbnail(image, thumbnail_size):
    radius = int(thumbnail_size / 2.0)
    center_x = radius
    center_y = radius

    thumbnail = cv2.resize(image, (thumbnail_size, thumbnail_size))
    cv2.circle(thumbnail, (center_x, center_y), radius, WHITE, thickness=1)

    return thumbnail


# to avoid having to create same thumbnail multiple times
def init_thumbnail_cache():
    cache = {}
    thumbnail_size_used_in_cache = None

    def get_thumbnail_f(person, thumbnail_size):
        nonlocal cache

        if thumbnail_size_used_in_cache is None or thumbnail_size_used_in_cache != thumbnail_size:
            cache = {}

        if len(cache) > 100:
            cache = {}

        if person.id not in cache:
            cache[person.id] = create_thumbnail(person.image, thumbnail_size)

        return cache[person.id]

    return get_thumbnail_f


# init thumbnail and mask cache to avoid expense calculations
get_thumbnail = init_thumbnail_cache()
get_round_mask = init_masks_cache()


def draw_thumbnail(frame, x, y, person):
    thumbnail_size = CONFIG["display"]["thumbnail_size"]

    thumbnail = get_thumbnail(person, thumbnail_size)
    mask = make_round_mask(thumbnail_size)
    copy_object_to_location(frame, thumbnail, x, y, mask)
