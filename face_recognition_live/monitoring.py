from collections import deque, defaultdict
from pathlib import Path
import pickle

MONITORING_PATH = Path("monitoring")
MONITORING_PATH.mkdir(exist_ok=True)


class MonitoringDatabase:
    def __init__(self, name):
        self.data = defaultdict(lambda: deque(maxlen=10000))
        self._file = (MONITORING_PATH / name).with_suffix(".pkl")
        self._storage_counter = 0

    def add(self, key, value):
        self.data[key].append(value)
        self._storage_counter += 1 % 1000
        if self._storage_counter % 10 == 0:
            self.store()

    def store(self):
        with self._file.open("wb") as f:
            pickle.dump(dict(self.data), f)
