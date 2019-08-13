import threading
import queue
from contextlib import contextmanager

from recognition.tasks import FaceDetection, FaceRecognition
from recognition.results import DetectedFaces
from recognition.models.face_detection import init_face_detector


class RecognitionThread(threading.Thread):
    def __init__(self, config, task_queue, result_queue):
        super(RecognitionThread, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_request = threading.Event()

        self.face_detector = init_face_detector(config)

    def run(self):
        for task in self._read_queue_until_quit():

            if isinstance(task, FaceDetection):
                face_locations = self.face_detector(task.image.data)
                self.result_queue.put(DetectedFaces(task.image, face_locations))

            elif isinstance(task, FaceRecognition):
                pass

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
