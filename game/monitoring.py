from functools import wraps
from datetime import datetime
from collections import deque
from pathlib import Path
import pickle
import logging

import numpy as np

MONITORING_PATH = Path("monitoring")


def calculate_runtime_in_seconds(start, end):
    delta = end - start
    return delta.seconds + delta.microseconds / 1000.0 / 1000.0


def monitor_runtime(f):
    logger = logging.getLogger(f.__name__)
    logger_avg = logging.getLogger(f.__name__+"_avg")
    history = deque(maxlen=10)
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


class ProbabilitiesMonitor:
    def __init__(self):
        self.data = []

    def add(self, matches):
        matches = [(str(m.face.id), m.distance) for m in matches]
        self.data.append((datetime.now(), matches))

        if len(self.data) == 10:
            self.store()
            self.data = []

    def store(self):
        now = datetime.now()

        directory = MONITORING_PATH / now.strftime("%Y%m%d-%H")
        directory.mkdir(exist_ok=True, parents=True)

        file = directory / (now.strftime("%Y%m%d-%H%M%S.%f") + ".pkl")
        with file.open("wb") as f:
            pickle.dump(self.data, f)



