from pathlib import Path

import dlib
import cv2


# todo freezes if negative locations -> filter out negative locations
def init_face_detector(model_dir: Path, model_type, scale):
    if model_type == "HOG":
        detector = init_face_detector_hog()
    elif model_type == "CNN":
        detector = init_face_detector_cnn(model_dir)
    else:
        raise NotImplementedError()

    def unscale(rect):
        return dlib.rectangle(int(rect.left() * 1.0 / scale), int(rect.top() * 1.0 / scale),
                              int(rect.right() * 1.0 / scale), int(rect.bottom() * 1.0 / scale))

    # todo too strict? rest of pipeline freezes for a bit if giving boxes outside the image
    def fully_inside_image(rect, image_width, image_height):
        return rect.left() > 0 and rect.right() < image_width and rect.top() > 0 and rect.bottom() < image_height

    def run(image):
        small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        result = detector(small_image)

        height, width = small_image.shape[0], small_image.shape[1]
        result = [r for r in result if fully_inside_image(r, width, height)]

        return [unscale(r) for r in result]

    return run


def init_face_detector_hog():
    detector = dlib.get_frontal_face_detector()

    def run(image):
        return detector(image[:, :, ::-1])

    return run


def init_face_detector_cnn(model_dir: Path):
    model_location = model_dir / "face_detection" / "mmod_human_face_detector.dat"
    detector = dlib.cnn_face_detection_model_v1(str(model_location))

    def run(image):
        return [r.rect for r in detector(image, 0)]  # todo check confidence

    return run
