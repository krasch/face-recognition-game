from collections import namedtuple
import datetime
import pickle
from pathlib import Path


from face_recognition_live.models.face_matching import find_best_match, MATCH_TYPE

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

    def is_frontal(self, location):
        return True

    def match(self, location, crop, features):
        match_type, matched_face = find_best_match(features, self.faces)
        print(match_type)

        if match_type == MATCH_TYPE.MATCH:
            return matched_face

        # was too near other faces, do not add to database just in case
        elif match_type == MATCH_TYPE.LIKELY_NO_MATCH:
            return None

        # not near any known faces, can safely add to database
        elif match_type == MATCH_TYPE.DEFINITELY_NO_MATCH:
            if self.is_frontal(None):
                self.add(features, crop)
            return None

    def match_all(self, locations, crops, features):
        matches = [self.match(l, c, f) for l, c, f in zip(locations, crops, features)]
        matches = [m for m in matches if m is not None]
        return matches

    def store(self):
        # only keep 1000 newest people
        self.faces = self.faces[-1000:]
        with self.database_file.open("wb") as f:
            pickle.dump(self.faces, f)





