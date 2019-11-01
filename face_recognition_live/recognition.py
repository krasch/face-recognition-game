import threading
import queue
from contextlib import contextmanager
import time
from abc import abstractmethod

import numpy as np

from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *
from face_recognition_live.database import FaceDatabase
from face_recognition_live.models import init_model_stack
from face_recognition_live.face import Face


class WorkerThread(threading.Thread):
    def __init__(self, task_queue, result_queue):
        super(WorkerThread, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_request = threading.Event()

    @abstractmethod
    def execute_task(self, task):
        pass

    def run(self):
        for task in self._read_queue_until_quit():
            result = self.execute_task(task)
            self.result_queue.put(result)

    def _read_queue_until_quit(self):
        while not self.stop_request.isSet():
            try:
                yield self.task_queue.get(True, 1.0 / 50.0)
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stop_request.set()
        super(WorkerThread, self).join(timeout)


class RecognitionThread(WorkerThread):
    def __init__(self, config, task_queue, result_queue):
        super(RecognitionThread, self).__init__(task_queue, result_queue)

        self.models = init_model_stack(config)
        self.face_database = FaceDatabase(config["recognition"]["database"]["location"])

    def execute_task(self, task):
        if isinstance(task, DetectFaces):
            bounding_boxes = self.models.detect_faces(task.image.data)
            faces = [Face(bounding_box=box) for box in bounding_boxes]
            return DetectionResult(task.image, faces)

        elif isinstance(task, CropFaces):
            for face in task.faces:
                face.landmarks = self.models.find_landmarks(task.image.data, face.bounding_box)
                face.frontal = self.models.is_frontal(face.bounding_box, face.landmarks)
                face.crop = self.models.crop(task.image.data, face.landmarks)
            return CroppingResult(task.image, task.faces)

        elif isinstance(task, ExtractFeatures):
            all_features = self.models.extract_features(np.array([face.crop for face in task.faces]))
            for face, features in zip(task.faces, all_features):
                face.features = features
            return FeatureExtractionResult(task.image, task.faces)

        elif isinstance(task, RecognizePeople):
            for face in task.faces:
                face.match = self.face_database.match(face.crop, face.features, face.frontal)
            return RecognitionResult(task.image.id, task.faces)

        elif isinstance(task, BackupFaceDatabase):
            self.face_database.store()

        else:
            raise NotImplementedError()


@contextmanager
def init_recognition(config, work_queue, results_queue):
    recognition = RecognitionThread(config, work_queue, results_queue)
    time.sleep(3)

    try:
        recognition.start()
        yield recognition
    finally:
        recognition.join()
