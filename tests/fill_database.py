import sys
sys.path.append(".")

import cv2
import numpy as np

from face_recognition_live.recognition.models import init_model_stack
from face_recognition_live.recognition.face import Face
from face_recognition_live.database import FaceDatabase

image = cv2.imread("tests/data/generated/2.png")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

models = init_model_stack()

box = models.detect_faces(image)[0]
thumbnail = image[box.top(): box.bottom(), box.left(): box.right(), :]
face = Face(box, thumbnail)
face.landmarks = models.find_landmarks(image, box)
face.frontal = models.is_frontal(box, face.landmarks)
face.crop = models.crop(image, face.landmarks)
face.features = models.extract_features(np.array([face.crop]))[0]

from profilehooks import profile
@profile
def bla():
    db = FaceDatabase("database.pkl")

    for i in range(1000):
        db.add_all([face])
    db.store()


bla()

