from pathlib import Path
from enum import Enum

import dlib
import cv2

from face_recognition_live.config import CONFIG


class DETECTION_MODEL(Enum):
    HOG = 1
    CNN = 2


def sort_bounding_boxes_by_size(boxes):
    if boxes is None or len(boxes) == 0:
        return boxes

    sizes = [(b.bottom() - b.top()) * (b.right() - b.left()) for b in boxes]
    boxes, _ = zip(*sorted(zip(boxes, sizes), key=lambda item: item[1], reverse=True))
    return boxes


def init_face_detector(model_type: DETECTION_MODEL, model_dir: Path):
    if model_type == DETECTION_MODEL.HOG:
        detector = init_face_detector_hog()
    elif model_type == DETECTION_MODEL.CNN:
        detector = init_face_detector_cnn(model_dir)
    else:
        raise NotImplementedError()

    def unscale(rect):
        scale = CONFIG["recognition"]["models"]["face_detection"]["scale"]
        return dlib.rectangle(int(rect.left() * 1.0 / scale), int(rect.top() * 1.0 / scale),
                              int(rect.right() * 1.0 / scale), int(rect.bottom() * 1.0 / scale))

    # todo too strict? rest of pipeline freezes for a bit if giving boxes outside the image
    def fully_inside_image(rect, image_width, image_height):
        return rect.left() > 0 and rect.right() < image_width and rect.top() > 0 and rect.bottom() < image_height

    def run(rgb_image):
        config = CONFIG["recognition"]["models"]["face_detection"]
        scale = config["scale"]
        small_image = cv2.resize(rgb_image, (0, 0), fx=scale, fy=scale)
        result = detector(small_image)

        height, width = small_image.shape[0], small_image.shape[1]
        result = [r for r in result if fully_inside_image(r, width, height)]
        result = [unscale(r) for r in result]
        result = sort_bounding_boxes_by_size(result)
        result = result[0: config["max_num_faces"]]

        return result

    return run


def init_face_detector_hog():
    detector = dlib.get_frontal_face_detector()

    def run(rgb_image):
        return detector(rgb_image)   # what does parameter "1" mean

    return run


def init_face_detector_cnn(model_dir: Path):
    model_location = model_dir / "face_detection" / "mmod_human_face_detector.dat"
    detector = dlib.cnn_face_detection_model_v1(str(model_location))

    def run(rgb_image):
        return [r.rect for r in detector(rgb_image, 0)]  # todo check confidence

    return run
