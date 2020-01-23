import cv2


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


get_stars = init_stars_cache()
