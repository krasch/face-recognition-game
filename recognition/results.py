class Result:
    def is_outdated(self, current_id, time_to_live):
        raise NotImplementedError()


class CameraImage(Result):
    def __init__(self, id_, image):
        self.id = id_
        self.data = image

    def is_outdated(self, current_id, time_to_live):
        return False


class DetectedFaces(Result):
    def __init__(self, original_image, face_locations):
        self.original_image = original_image
        self.face_locations = face_locations

    def is_outdated(self, current_id, time_to_live):
        age = current_id - self.original_image.id
        return age > time_to_live


class RecognizedPersons(Result):
    def __init__(self, original_image_id, persons):
        self.original_image_id = original_image_id
        self.persons = persons

    def is_outdated(self, current_id, time_to_live):
        age = current_id - self.original_image_id
        return age > time_to_live
