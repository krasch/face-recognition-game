from functools import wraps
from datetime import datetime
from collections import deque, defaultdict
from pathlib import Path
import pickle
import logging

import numpy as np

MONITORING_PATH = Path("monitoring")
MONITORING_PATH.mkdir(exist_ok=True)

HISTORY_LENGTH = 10


def calculate_runtime_in_seconds(start, end):
    delta = end - start
    return delta.seconds + delta.microseconds /1000.0 / 1000.0


def monitor_runtime(f):
    logger = logging.getLogger(f.__name__)
    logger_avg = logging.getLogger(f.__name__+"_avg")
    history = deque(maxlen=HISTORY_LENGTH)
    counter = 0

    @wraps(f)
    def decorated(*args, **kwargs):
        # execute the function
        start = datetime.now()
        result = f(*args, **kwargs)
        end = datetime.now()

        # handle the statistics
        runtime = calculate_runtime_in_seconds(start, end)
        history.append(runtime)

        # print debugging info
        logger.debug("runtime={}".format(np.round(runtime, 4)))
        nonlocal counter
        counter = (counter + 1) % 10
        if len(history) == 10 and counter == 0:
            logger_avg.debug("runtime={}".format(np.round(np.mean(history), 4)))

        return result
    return decorated


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



