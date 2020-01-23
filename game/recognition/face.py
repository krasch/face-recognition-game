class Face:
    def __init__(self, bounding_box, thumbnail):
        self.bounding_box = bounding_box
        self.thumbnail = thumbnail
        self.landmarks = None
        self.crop = None
        self.frontal = None
        self.features = None
        self.matches = None
        self.debug = {}
