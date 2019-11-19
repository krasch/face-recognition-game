from PIL import ImageFont, ImageDraw, Image
import numpy as np

THREE_STARS = u"★★★"
TWO_STARS = u"★★☆"
ONE_STAR = u"★☆☆"
NO_STARS = u"☆☆☆"

STARS = {0: NO_STARS, 1: ONE_STAR, 2: TWO_STARS, 3: THREE_STARS}

font = ImageFont.truetype("OpenSansEmoji.ttf", 40)


def draw_stars(num_stars):
    image = np.ones((30, 90, 4), np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text((0, -12), STARS[num_stars], font=font)
    image.save("stars_{}.png".format(num_stars))

draw_stars(0)
draw_stars(1)
draw_stars(2)
draw_stars(3)

