from pathlib import Path
from threading import Lock
from collections import MutableMapping
from copy import deepcopy

import yaml


class Config(MutableMapping):
    """
    Config that behaves exactly like a (read-only) dictionary, but that can be re-read from file if necessary, in a
    thread-sage manner
    """
    config = None

    def __init__(self, config_file):
        self.config_file = Path(config_file)
        # self._lock = Lock()
        self.reload()

    def __getitem__(self, key):
        #self._lock.acquire()
        value = self.config[key]
        #self._lock.release()
        return value

    def reload(self):
        print("Reloading config")
        #self._lock.acquire()
        self.config = yaml.safe_load(self.config_file.read_text())
        #self._lock.release()

    def toggle_debug(self):
        config_copy = deepcopy(self.config)
        config_copy["display"]["debug"] = not config_copy["display"]["debug"]
        self.config = config_copy

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.config)


CONFIG = Config("config.yaml")


