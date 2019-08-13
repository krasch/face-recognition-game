import threading
import queue

import dlib
import cv2


# todo freezes if negative locations -> should be no negative locations
def init_face_detector_cnn():
    detector = dlib.cnn_face_detection_model_v1("models/face_detection/mmod_human_face_detector.dat")
    # scale = FACE_DETECTION_INPUT_WIDTH / DISPLAY_WIDTH
    scale = 0.25

    def unscale(result):
        return dlib.rectangle(int(result.left()*1.0/scale), int(result.top()*1.0/scale),
                              int(result.right()*1.0/scale), int(result.bottom()*1.0/scale))

    def run(frame):
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        return [unscale(r.rect) for r in detector(small_frame, 0)]  # todo check confidence

    return run


class RecognitionThread(threading.Thread):
    def __init__(self, input_queue, output_queue):
        super(RecognitionThread, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stoprequest = threading.Event()

        self.face_detector = init_face_detector_cnn()

    def run(self):
        while not self.stoprequest.isSet():
            try:
                _, image = self.input_queue.get(True, 1.0/15.0)
                result = self.face_detector(image)
                self.output_queue.put(("face", result))
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stoprequest.set()
        super(RecognitionThread, self).join(timeout)
