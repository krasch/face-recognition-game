from collections import namedtuple
import datetime
import pickle
from pathlib import Path


from face_recognition_live.models.matching import find_best_match, MATCH_TYPE

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

    def match(self, crop, features, is_frontal):
        match_type, matched_face = find_best_match(features, self.faces)

        if match_type == MATCH_TYPE.MATCH:
            return matched_face

        # was too near other faces, do not add to database just in case
        elif match_type == MATCH_TYPE.LIKELY_NO_MATCH:
            return None

        # not near any known faces, can safely add to database
        elif match_type == MATCH_TYPE.DEFINITELY_NO_MATCH:
            if is_frontal:
                print("Adding face to database")
                self.add(features, crop)
            return None

    def store(self):
        # only keep 1000 newest people
        self.faces = self.faces[-1000:]
        with self.database_file.open("wb") as f:
            pickle.dump(self.faces, f)





