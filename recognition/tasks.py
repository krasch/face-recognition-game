class Task:
    pass


class FaceDetection(Task):
    def __init__(self, image):
        self.image = image


class FaceRecognition(Task):
    def __init__(self, image, face_locations):
        self.image = image
        self.face_locations = face_locations
