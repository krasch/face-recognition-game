import threading
import queue
from contextlib import contextmanager
import time

import numpy as np

from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *
from face_recognition_live.database import FaceDatabase
from face_recognition_live.models import init_model_stack
from face_recognition_live.face import Face


class RecognitionThread(threading.Thread):
    def __init__(self, config, task_queue, result_queue):
        super(RecognitionThread, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_request = threading.Event()

        self.models = init_model_stack(config)
        self.face_database = FaceDatabase(config["recognition"]["database"]["location"])

    def run(self):
        for task in self._read_queue_until_quit():

            if isinstance(task, RecognizeFaces):
                bounding_boxes = self.models.detect_faces(task.image.data)
                faces = [Face(bounding_box) for bounding_box in bounding_boxes]

                # todo parallelize?
                for face in faces:
                    face.landmarks = self.models.find_landmarks(task.image.data, face.bounding_box)
                    face.frontal, face.debug["frontal"] = self.models.is_frontal(task.image.data, face.bounding_box, face.landmarks)
                    face.crop = self.models.crop(task.image.data, face.landmarks)

                # do all faces at same time, much faster
                all_features = self.models.extract_features(np.array([face.crop for face in faces]))
                for face, features in zip(faces, all_features):
                    face.features = features

                # todo parallelize?
                for face in faces:
                    face.match = self.face_database.match(face.crop, face.features, face.frontal)

                self.result_queue.put(RecognitionResult(task.image.id, faces))

            elif isinstance(task, BackupFaceDatabase):
                self.face_database.store()

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
