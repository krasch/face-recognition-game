from collections import namedtuple
import datetime

import numpy as np


StoredFace = namedtuple("StoredFace", ["timestamp", "features", "image"])


class FaceDatabase:
    def __init__(self):
        self.faces = []

    def is_empty(self):
        return len(self.faces) == 0

    def add(self, features, image):
        timestamp = datetime.datetime.now()
        self.faces.append(StoredFace(timestamp, features, image))

    def is_frontal(self, location):
        return True

    def match(self, location, crop, features):
        if self.is_empty():
            if self.is_frontal(None):
                self.add(features, crop)
            return None

        distances = np.array([np.linalg.norm(features - saved_face.features) for saved_face in self.faces])
        best_match = np.argmin(distances)

        if distances[best_match] < 0.6:
            return self.faces[best_match]
        else:
            if self.is_frontal(None) and distances[best_match] > 2.0:
                self.add(features, crop)
            return None
    
    def match_all(self, locations, crops, features):
        return [self.match(l, c, f) for l, c, f in zip(locations, crops, features)]

    def store(self):
        pass
