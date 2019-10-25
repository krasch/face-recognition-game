class Face:
    def __init__(self, bounding_box):
        self.bounding_box = bounding_box
        self.landmarks = None
        self.crop = None
        self.frontal = None
        self.features = None
        self.match = None
        self.debug = {}
