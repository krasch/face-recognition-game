from uuid import uuid4
from collections import namedtuple
import datetime
import pickle
from pathlib import Path

from face_recognition_live.monitoring import monitor_runtime


StoredFace = namedtuple("StoredFace", ["group_id", "timestamp", "features", "thumbnail"])


class FaceDatabase:

    def __init__(self, database_file):
        self.database_file = Path(database_file)

        if self.database_file.exists():
            self.faces = pickle.loads(self.database_file.read_bytes())
        else:
            self.faces = []

    def add_all(self, faces):
        group_id = uuid4()
        for face in faces:
            self._add(group_id, face.features, face.thumbnail)

    def _add(self, group_id, features, thumbnail):
        timestamp = datetime.datetime.now()
        self.faces.append(StoredFace(group_id, timestamp, features, thumbnail))

    def remove_most_recent(self):
        if len(self.faces) == 0:
            return []

        most_recent_group_id = self.faces[-1].group_id
        to_remove = [i for i, face in enumerate(self.faces) if face.group_id == most_recent_group_id]
        removed_thumbnails = [self.faces[i].thumbnail for i in to_remove]
        self.faces = [face for i, face in enumerate(self.faces) if i not in to_remove]
        return removed_thumbnails

    @monitor_runtime
    def store(self):
        # only keep 1000 newest people
        self.faces = self.faces[-1000:]
        with self.database_file.open("wb") as f:
            pickle.dump(self.faces, f)





