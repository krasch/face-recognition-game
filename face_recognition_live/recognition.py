import threading
import queue
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path
import time

from face_recognition_live.tasks import RecognizeFaces, BackupFaceDatabase
from face_recognition_live.results import Faces
from face_recognition_live.models.face_detection import init_face_detector
from face_recognition_live.models.face_cropping import init_face_cropper
from face_recognition_live.models.feature_extraction import init_feature_extractor
from face_recognition_live.models.face_matching import FaceDatabase


Face = namedtuple("Face", ["bounding_box", "match"])


def init_model_stack(config):
    model_dir = Path(config["recognition"]["models"]["directory"])

    if config["recognition"]["models"]["stack"] == "CNN+dlibcrop+openface":
        scale = config["recognition"]["models"]["face_detection"]["scale"]
        face_detector = init_face_detector(model_dir, "CNN", scale)
        face_cropper = init_face_cropper(model_dir)
        feature_extractor = init_feature_extractor(model_dir)
        return face_detector, face_cropper, feature_extractor
    else:
        raise NotImplementedError()


class RecognitionThread(threading.Thread):
    def __init__(self, config, task_queue, result_queue):
        super(RecognitionThread, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_request = threading.Event()

        # load all the models
        self.face_detector, self.face_cropper, self.feature_extractor = init_model_stack(config)
        self.face_database = FaceDatabase()

    def run(self):
        for task in self._read_queue_until_quit():

            if isinstance(task, RecognizeFaces):
                # todo break this down into several tasks?
                face_locations = self.face_detector(task.image.data)
                print(face_locations)

                if face_locations:

                    crops = self.face_cropper(task.image.data, face_locations)
                    features = self.feature_extractor(crops)
                    matches = self.face_database.match_all(face_locations, crops, features)

                    faces = [Face(location, match) for location, match in zip(face_locations, matches)]
                    self.result_queue.put(Faces(task.image.id, faces))

                else:
                    self.result_queue.put(Faces(task.image.id, []))

            elif isinstance(task, BackupFaceDatabase):
                print("Should backup face database")  # todo

            else:
                raise NotImplementedError()

    def _read_queue_until_quit(self):
        while not self.stop_request.isSet():
            try:
                yield self.task_queue.get(True, 1.0 / 50.0)
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stop_request.set()
        super(RecognitionThread, self).join(timeout)


@contextmanager
def init_recognition(config, work_queue, results_queue):
    recognition = RecognitionThread(config, work_queue, results_queue)
    time.sleep(3)

    try:
        recognition.start()
        yield recognition
    finally:
        recognition.join()
