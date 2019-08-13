from contextlib import contextmanager
import time
import queue

import cv2
import dlib
import numpy as np

from camera import CameraThread
from recognition import RecognitionThread


# CAMERA SETTINGS
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FLIP = 0
ZOOM = 1.00
FRAMERATE = 15

# SCREEN SETTINGS
# todo check screen ratio when fullscreen selected
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FULLSIZE=False

# MODEL SETTINGS
FACE_DETECTION_MODEL = "HOG" # CNN
FACE_DETECTION_INPUT_WIDTH = 1280/4





@contextmanager
def init_display(fullsize):
    name = "window"
    if fullsize:
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow("window")
   
    try:
        yield name
    finally:
        cv2.destroyWindow(name)          


def init_face_detector_hog():
    # todo scale
    detector = dlib.get_frontal_face_detector()
    
    def run(frame):
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        return detector(small_frame[:, :, ::-1])

    return run


def init_face_cropper():
    landmark_model = dlib.shape_predictor("models/face_cropping/shape_predictor_5_face_landmarks.dat")

    def run(frame, face):
        shape = landmark_model(frame[:, :, ::-1], face) 
        return dlib.get_face_chip(frame, shape)[:, :, ::-1] # todo 96

    return run


def init_feature_extractor():
    feature_model = cv2.dnn.readNetFromTorch("models/feature_extraction/nn4.small2.v1.t7")

    def run(face):
        blob = cv2.dnn.blobFromImage(face, 1.0/255, (96,96), swapRB=False, crop=False) # todo resize should not be necessary
        feature_model.setInput(blob)
        return feature_model.forward()

    return run


def init_face_matching():
    num_existing = 10
    face_database = list(np.random.rand(num_existing, 128))

    def run(face):
        # todo proper init
        if len(face_database) == num_existing:
            face_database.append(face)
        distances = [np.linalg.norm(face-saved_face) for saved_face in face_database]
        print(distances)
        matches = [d for d in distances if d<1.0]
        return len(matches) > 0

    return run


#@profile
def main():
    detect_faces = init_face_detector_cnn()
    crop_faces = init_face_cropper()
    extract_features = init_feature_extractor()
    match_faces = init_face_matching()

    time.sleep(3)  # to get model loading to finish

    with init_display(FULLSIZE) as display_name:
        with open_stream() as stream:
            for i, frame in enumerate(read_stream_until_quit(stream)):  # todo can enumerate overflow?
                cv2.imshow(display_name, frame)
                print(frame.shape)

                face_locations = detect_faces(frame)
                if not face_locations:
                    continue

                for face in face_locations:
                    cropped = crop_faces(frame, face)  # openface has its own alignment, seems to use 96 points
                    features = extract_features(cropped)[0]
                    matched = match_faces(features)
                    if matched:
                       color = (0, 0, 255)
                    else:
                       color = (255, 0, 0)
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

                cv2.imshow(display_name, frame)


def main2():
    work = queue.Queue()
    results = queue.Queue()

    recognition = RecognitionThread(work, results)
    recognition.start()
    time.sleep(3)

    camera = CameraThread(results)
    camera.start()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            t, data = results.get(True, 1.0/15.0)
            if t == "image":
                cv2.imshow("test", data)
                work.put(("image", data))
            else:
                print(data)
        except queue.Empty:
            continue

        #result = results.get()
        #print(result.shape)

        #cv2.imshow("test", frame)

    camera.join()
    recognition.join()


main2()

