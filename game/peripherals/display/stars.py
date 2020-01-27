import cv2

from game.config import CONFIG
from game.recognition.models.matching import MatchQuality
from game.peripherals.display.util import copy_object_to_location


def init_stars_cache():
    stars_original = {0: cv2.imread("images/stars_0.png", cv2.IMREAD_UNCHANGED),
                      1: cv2.imread("images/stars_1.png", cv2.IMREAD_UNCHANGED),
                      2: cv2.imread("images/stars_2.png", cv2.IMREAD_UNCHANGED),
                      3: cv2.imread("images/stars_3.png", cv2.IMREAD_UNCHANGED)}

    stars_resized = {0: {}, 1: {}, 2: {}, 3: {}}

    def get_stars_f(num_stars, stars_height):
        if stars_height not in stars_resized[num_stars]:
            s = cv2.resize(stars_original[num_stars], (stars_height * 3, stars_height))
            stars_resized[num_stars][stars_height] = s

        return stars_resized[num_stars][stars_height]

    return get_stars_f


def get_num_stars(match_quality):
    if match_quality == MatchQuality.EXCELLENT:
        return 3
    elif match_quality == MatchQuality.GOOD:
        return 2
    elif match_quality == MatchQuality.POOR:
        return 1
    else:
        return 0


# init stars cache to avoid expense calculations
get_stars = init_stars_cache()


def draw_stars(frame, x, y, match_quality):
    stars_height = CONFIG["display"]["stars_height"]

    num_stars = get_num_stars(match_quality)
    stars = get_stars(num_stars, stars_height)
    alpha = stars[:, :, 3:4].astype(float) / 255

    copy_object_to_location(frame, stars, x, y, 1 - alpha)
