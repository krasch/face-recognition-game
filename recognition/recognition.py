import threading
import queue
from contextlib import contextmanager

import dlib
import cv2

from recognition.types import TaskTypes, ResultTypes, Result


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
    def __init__(self, config, task_queue, result_queue):
        super(RecognitionThread, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_request = threading.Event()

        self.face_detector = init_face_detector_cnn()

    def run(self):
        for task in self._read_queue_until_quit():

            if task.type == TaskTypes.DETECT_FACES:
                result = self.face_detector(task.data)
                self.result_queue.put(Result(ResultTypes.FACE_DETECTIONS, (task.data, result)))

            elif task.type == TaskTypes.RECOGNIZE_PERSON:
                image, face = task.data

            else:
                raise NotImplementedError()

    def _read_queue_until_quit(self):
        while not self.stop_request.isSet():
            try:
                yield self.task_queue.get(True, 1.0 / 15.0)
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stop_request.set()
        super(RecognitionThread, self).join(timeout)


@contextmanager
def init_recognition(config, work_queue, results_queue):
    recognition = RecognitionThread(config, work_queue, results_queue)
    # time.sleep(3)  # give model loading time to finish

    try:
        recognition.start()
        yield recognition
    finally:
        recognition.join()
