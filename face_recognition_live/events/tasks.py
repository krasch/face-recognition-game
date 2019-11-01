class DetectFaces:
    def __init__(self, image):
        self.image = image


class CropFaces:
    def __init__(self, image, faces):
        self.image = image
        self.faces = faces


class ExtractFeatures:
    def __init__(self, image, faces):
        self.image = image
        self.faces = faces


class RecognizePeople:
    def __init__(self, image, faces):
        self.image = image
        self.faces = faces


class BackupFaceDatabase:
    def __init__(self):
        pass
