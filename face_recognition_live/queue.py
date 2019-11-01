import queue
from datetime import datetime

from face_recognition_live.monitoring import MonitoringDatabase


class MonitoredQueue(queue.Queue):
    def __init__(self, name, maxsize=0):
        super().__init__(maxsize)
        self.monitoring = MonitoringDatabase(name)

    def get(self, block=True, timeout=None):
        get_ts = datetime.now()
        put_ts, obj = super().get(block, timeout)
        wait_time = (get_ts - put_ts).microseconds
        self.monitoring.add(type(obj), wait_time)
        return obj

    def put(self, obj, block=True, timeout=None):
        put_ts = datetime.now()
        return super().put((put_ts, obj), block, timeout)
