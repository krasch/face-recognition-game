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
from face_recognition_live.recognition.models.matching import MATCH_TYPE
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
                yield self.task_queue.get(True, 1.0 / 50.0)
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
                match_result = self.face_database.match(face.features)
                self._monitoring.add("distances", match_result.distances)

                # set match
                if match_result.match_type == MATCH_TYPE.MATCH:
                    face.match = match_result

                # was too near other faces, do not add to database just in case
                elif match_result.match_type == MATCH_TYPE.LIKELY_NO_MATCH:
                    pass

                # not near any known faces, can safely add to database
                elif match_result.match_type == MATCH_TYPE.DEFINITELY_NO_MATCH:
                    if face.frontal.is_frontal:
                        print("Adding face to database")
                        self.face_database.add(face.features, face.crop)

            return RecognitionResult(task.image.id, task.faces)

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
