import threading
import queue
from contextlib import contextmanager
import time
from abc import abstractmethod
from datetime import datetime

import numpy as np

from face_recognition_live.config import CONFIG
from face_recognition_live.events.tasks import *
from face_recognition_live.events.results import *
from face_recognition_live.database import FaceDatabase
from face_recognition_live.recognition.models import init_model_stack
from face_recognition_live.recognition.models.matching import MatchResultOverall
from face_recognition_live.recognition.face import Face
from face_recognition_live.monitoring import MonitoringDatabase


class WorkerThread(threading.Thread):
    def __init__(self, task_queue, result_queue):
        super(WorkerThread, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_request = threading.Event()

        self._runtime_monitoring = MonitoringDatabase("recognition_runtime")

    @abstractmethod
    def execute_task(self, task):
        pass

    def run(self):
        for task in self._read_queue_until_quit():
            start = datetime.now()
            result = self.execute_task(task)
            end = datetime.now()
            self._runtime_monitoring.add(type(task), (end - start).microseconds)

            if result:
                self.result_queue.put(result)

    def _read_queue_until_quit(self):
        while not self.stop_request.isSet():
            try:
                yield self.task_queue.get(True, 1.0)
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stop_request.set()
        super(WorkerThread, self).join(timeout)


class RecognitionThread(WorkerThread):
    def __init__(self, task_queue, result_queue):
        super(RecognitionThread, self).__init__(task_queue, result_queue)

        self.models = init_model_stack()
        self.face_database = FaceDatabase(CONFIG["recognition"]["database"]["location"])
        self._monitoring = MonitoringDatabase("recognition")

    def execute_task(self, task):
        if isinstance(task, DetectFaces):
            bounding_boxes = self.models.detect_faces(task.image.data)
            faces = [Face(bounding_box=box) for box in bounding_boxes]
            self._monitoring.add("face_count", len(faces))

            for face in faces:
                face.landmarks = self.models.find_landmarks(task.image.data, face.bounding_box)
                face.frontal = self.models.is_frontal(face.bounding_box, face.landmarks)
                face.crop = self.models.crop(task.image.data, face.landmarks)

            all_features = self.models.extract_features(np.array([face.crop for face in faces]))
            for face, features in zip(faces, all_features):
                face.features = features

            for face in faces:
                match_status, matches = self.models.match_faces(face.features, self.face_database.faces)

                if match_status == MatchResultOverall.NEW_FACE:
                    if face.frontal.is_frontal:
                        print("Adding face to database")
                        self.face_database.add(face.features, face.crop)
                else:
                    self._monitoring.add("distances", [m.distance for m in matches])  #  todo move to matching
                    face.matches = matches

            return RecognitionResult(task.image.id, faces)

        elif isinstance(task, BackupFaceDatabase):
            self.face_database.store()

        else:
            raise NotImplementedError()


@contextmanager
def init_recognition(work_queue, results_queue):
    recognition = RecognitionThread(work_queue, results_queue)
    time.sleep(3)

    try:
        recognition.start()
        yield recognition
    finally:
        recognition.join()
