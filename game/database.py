from uuid import uuid4
from collections import namedtuple
import datetime
import pickle
from pathlib import Path

from game.monitoring import monitor_runtime


class StoredFace:
    def __init__(self, group_id, features, thumbnail):
        self.group_id = group_id
        self.features = features
        self.image = thumbnail
        self.timestamp = datetime.datetime.now()
        self.id = uuid4()

    def belongs_to_group(self, group_id):
        return self.group_id == group_id


class FaceDatabase:

    def __init__(self, database_file):
        self.database_file = Path(database_file)

        if self.database_file.exists():
            self.faces = pickle.loads(self.database_file.read_bytes())
        else:
            self.faces = []

    def add_all(self, faces):
        group_id = uuid4()
        registered = []
        for face in faces:
            face = StoredFace(group_id, face.features, face.thumbnail)
            self.faces.append(face)
            registered.append(face)
        return registered

    def remove_most_recent(self):
        if len(self.faces) == 0:
            return []

        most_recent_group_id = self.faces[-1].group_id
        to_remove = [i for i, face in enumerate(self.faces) if face.belongs_to_group(most_recent_group_id)]
        removed_faces = [self.faces[i] for i in to_remove]
        self.faces = [face for i, face in enumerate(self.faces) if i not in to_remove]
        return removed_faces

    @monitor_runtime
    def store(self):
        # only keep 1000 newest people
        self.faces = self.faces[-1000:]
        with self.database_file.open("wb") as f:
            pickle.dump(self.faces, f)





