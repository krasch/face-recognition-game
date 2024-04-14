import sys
from pathlib import Path
from copy import deepcopy

import yaml

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


class Config(MutableMapping):
    """
    Config that behaves exactly like a (read-only) dictionary, but that can be re-read from file if necessary, in a
    thread-safe manner
    """
    config = None

    def __init__(self, config_file):
        self.config_file = Path(config_file)
        self.reload()

    def __getitem__(self, key):
        value = self.config[key]
        return value

    def reload(self):
        print("Reloading config")
        self.config = yaml.safe_load(self.config_file.read_text())

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


