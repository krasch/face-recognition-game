import threading
from multiprocessing import Process
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
from face_recognition_live.monitoring import MonitoringDatabase, monitor_runtime


class WorkerProcess(Process):
    def __init__(self, task_queue, result_queue):
        super(WorkerProcess, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

        self._stop = False

        self._runtime_monitoring = MonitoringDatabase("recognition_runtime")

    @abstractmethod
    def execute_task(self, task):
        pass

    def get_task(self):
        if not self.task_queue.empty():
            return True, self.task_queue.get()
        else:
            return False, None

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            ret, task = self.get_task()
            if ret:
                result = self.execute_task(task)
                if result:
                    self.result_queue.put(result)


class RecognitionProcess(WorkerProcess):
    def __init__(self, task_queue, result_queue):
        super(RecognitionProcess, self).__init__(task_queue, result_queue)

        self.models = init_model_stack()
        self.face_database = FaceDatabase(CONFIG["recognition"]["database"]["location"])
        self._monitoring = MonitoringDatabase("recognition")

    def execute_task(self, task):
        if isinstance(task, DetectFaces):
            faces = self.__detect_faces(task.image.data)
            if len(faces) > 0:
                self._crop(task.image.data, faces)
                self._extract_features(faces)
                self._find_matches(faces)
            return RecognitionResult(task.image.id, faces)

        elif isinstance(task, BackupFaceDatabase):
            self.face_database.store()

        else:
            raise NotImplementedError()

    @monitor_runtime
    def __detect_faces(self, image):
        bounding_boxes = self.models.detect_faces(image)
        faces = [Face(bounding_box=box) for box in bounding_boxes]
        #self._monitoring.add("face_count", len(faces))
        return faces

    @monitor_runtime
    def _crop(self, image, faces):
        for face in faces:
            face.landmarks = self.models.find_landmarks(image, face.bounding_box)
            face.frontal = self.models.is_frontal(face.bounding_box, face.landmarks)
            face.crop = self.models.crop(image, face.landmarks)

    @monitor_runtime
    def _extract_features(self, faces):
        all_features = self.models.extract_features(np.array([face.crop for face in faces]))
        for face, features in zip(faces, all_features):
            face.features = features

    @monitor_runtime
    def _find_matches(self, faces):
        for face in faces:
            match_status, matches = self.models.match_faces(face.features, self.face_database.faces)

            if match_status == MatchResultOverall.NEW_FACE:
                if face.frontal.is_frontal:
                    print("Adding face to database")
                    self.face_database.add(face.features, face.crop)
            else:
                self._monitoring.add("distances", [m.distance for m in matches])  # todo move to matching
                face.matches = matches


@contextmanager
def init_recognition(work_queue, results_queue):
    recognition = RecognitionProcess(work_queue, results_queue)
    time.sleep(3)

    try:
        recognition.start()
        yield recognition
    finally:
        recognition.stop()
