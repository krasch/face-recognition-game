class Result:
    def __init__(self, image_id):
        self.id = image_id

    def is_outdated(self, current_id, time_to_live):
        age = current_id - self.id
        return age > time_to_live


class CameraImage(Result):
    def __init__(self, image_id, image):
        super().__init__(image_id)
        self.data = image


class RecognitionResult(Result):
    def __init__(self, image_id, faces):
        super().__init__(image_id)
        self.faces = faces


class Error:
    pass
