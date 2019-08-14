import numpy as np


class FaceDatabase:
    def __init__(self):
        self.faces = []

    def add(self, face):
        self.faces.append(face)

    def match(self, face):
        distances = [np.linalg.norm(face - saved_face) for saved_face in self.faces]
        matches = [d for d in distances if d < 1.0]
        is_match = len(matches) > 0
        if not is_match:
            self.add(face)
        return is_match
    
    def match_all(self, faces):
        return [self.match(face) for face in faces]

    def store(self):
        pass
