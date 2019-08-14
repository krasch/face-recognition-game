
class CameraImage:
    def __init__(self, id_, image):
        self.id = id_
        self.data = image


class Faces:
    def __init__(self, image_id, persons):
        self.image_id = image_id
        self.persons = persons

    def is_outdated(self, current_id, time_to_live):
        age = current_id - self.image_id
        return age > time_to_live
