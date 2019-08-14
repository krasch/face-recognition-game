import numpy as np
import cv2


def assert_face_detections_equal(detection1, detection2):
    assert len(detection1) == len(detection2)

    sort_order = lambda f: (f.left(), f.top(), f.right(), f.bottom())
    detection1 = sorted(detection1, key=sort_order)
    detection2 = sorted(detection1, key=sort_order)

    for i in range(len(detection1)):
        face1 = detection1[i]
        face2 = detection2[i]
        assert face1.left() == face2.left()
        assert face1.top() == face2.top()
        assert face1.right() == face2.right()
        assert face1.bottom() == face2.bottom()


def shift_image(image, x_shift, y_shift):
    height, width = image.shape[0], image.shape[1]
    assert abs(x_shift) < width
    assert abs(y_shift) < height
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted = cv2.warpAffine(image, M, (width, height))
    return shifted
