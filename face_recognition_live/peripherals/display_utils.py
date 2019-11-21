import cv2
import numpy as np

from face_recognition_live.monitoring import monitor_runtime


WHITE = (255, 255, 255)


def make_round_mask(diameter):
    radius = int(diameter / 2.0)
    center_x = radius
    center_y = radius

    mask = np.zeros((diameter, diameter, 1), np.uint8)
    mask.fill(255)
    cv2.circle(mask, (center_x, center_y), radius, 3, thickness=-1)
    return mask / 255.0


def create_thumbnail(image, thumbnail_size):
    radius = int(thumbnail_size / 2.0)
    center_x = radius
    center_y = radius

    thumbnail = cv2.resize(image, (thumbnail_size, thumbnail_size))
    cv2.circle(thumbnail, (center_x, center_y), radius, WHITE, thickness=1)

    return thumbnail


def init_thumbnail_cache():
    cache = {}
    thumbnail_size_used_in_cache = None

    def get_thumbnail(person, thumbnail_size):
        nonlocal cache

        if thumbnail_size_used_in_cache is None or thumbnail_size_used_in_cache != thumbnail_size:
            cache = {}

        if len(cache) > 100:
            cache = {}

        if person.id not in cache:
            cache[person.id] = create_thumbnail(person.image, thumbnail_size)

        return cache[person.id]

    return get_thumbnail


def init_masks_cache():
    masks = {}

    def get_mask(diameter):
        if diameter not in masks:
            masks[diameter] = make_round_mask(diameter)
        return masks[diameter]

    return get_mask


def init_stars_cache():
    stars_original = {0: cv2.imread("images/stars_0.png", cv2.IMREAD_UNCHANGED),
                      1: cv2.imread("images/stars_1.png", cv2.IMREAD_UNCHANGED),
                      2: cv2.imread("images/stars_2.png", cv2.IMREAD_UNCHANGED),
                      3: cv2.imread("images/stars_3.png", cv2.IMREAD_UNCHANGED)}

    stars_resized = {0: {}, 1: {}, 2: {}, 3: {}}

    def get_stars(num_stars, stars_height):
        if stars_height not in stars_resized[num_stars]:
            s = cv2.resize(stars_original[num_stars], (stars_height * 3, stars_height))
            stars_resized[num_stars][stars_height] = s

        return stars_resized[num_stars][stars_height]

    return get_stars


def copy_object_to_location(frame, object_to_copy, x, y, mask):
    frame_height, frame_width, _ = frame.shape
    object_height, object_width, _ = object_to_copy.shape

    if x < 0 or x + object_width > frame_width or y < 0 or y + object_height > frame_height:
        return

    original = frame[y: y + object_height, x: x + object_width, 0:3] * mask
    new = object_to_copy[:, :, 0:3] * (1-mask)
    frame[y: y + object_height, x: x + object_width, 0:3] = original + new

"""
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
"""

get_thumbnail = init_thumbnail_cache()
get_round_mask = init_masks_cache()
get_stars = init_stars_cache()
