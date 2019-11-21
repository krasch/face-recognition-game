from datetime import datetime


class Result:
    def __init__(self):
        self.timestamp = datetime.now()

    def is_outdated(self, time_to_live):
        now = datetime.now()
        age = now - self.timestamp
        age = age.seconds * 1000.0 + age.microseconds / 1000.0
        return age > time_to_live


class CameraImage(Result):
    def __init__(self, image_id, image):
        super().__init__()
        self.id = image_id
        self.data = image


class RecognitionResult(Result):
    def __init__(self, faces):
        super().__init__()
        self.faces = faces


class RegistrationResult(Result):
    def __init__(self, persons):
        super().__init__()
        self.persons = persons


class UnregistrationResult(Result):
    def __init__(self, persons):
        super().__init__()
        self.persons = persons


class Error:
    pass
