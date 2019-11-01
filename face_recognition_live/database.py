from collections import namedtuple
import datetime
import pickle
from pathlib import Path


from face_recognition_live.recognition.models.matching import find_best_match

StoredFace = namedtuple("StoredFace", ["timestamp", "features", "image"])


class FaceDatabase:
    def __init__(self, database_file):
        self.database_file = Path(database_file)

        if self.database_file.exists():
            self.faces = pickle.loads(self.database_file.read_bytes())
        else:
            self.faces = []

    def add(self, features, image):
        timestamp = datetime.datetime.now()
        self.faces.append(StoredFace(timestamp, features, image))

    def match(self, features):
        return find_best_match(features, self.faces)

    def store(self):
        # only keep 1000 newest people
        self.faces = self.faces[-1000:]
        with self.database_file.open("wb") as f:
            pickle.dump(self.faces, f)





