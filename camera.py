from contextlib import contextmanager
from profilehooks import profile
import time

import cv2
import dlib
import numpy as np

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


def get_jetson_gstreamer_source():
    # calculate crop window
    # somewhat unintuitively gstreamer will calculate it such that the cropped window still has CAPTURE_HEIGHT*CAPTURE_WIDTH
    assert ZOOM >= 1
    crop_left = int((CAPTURE_WIDTH - CAPTURE_WIDTH/ZOOM)/2.0)
    crop_right = CAPTURE_WIDTH - crop_left
    crop_top = int((CAPTURE_HEIGHT - CAPTURE_HEIGHT/ZOOM)/2.0)
    crop_bottom = CAPTURE_HEIGHT - crop_top

    return (
            f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            f'width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}, ' +
            f'format=(string)NV12, framerate=(fraction){FRAMERATE}/1 ! ' +
            # f'nvvidconv flip-method={flip_method} ! ' +
            f'nvvidconv flip-method={FLIP} left={crop_left} right={crop_right} top={crop_top} bottom={crop_bottom} ! ' +
            f'video/x-raw, width=(int){DISPLAY_WIDTH}, height=(int){DISPLAY_HEIGHT}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

@contextmanager
def open_stream():
    video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    try:
        yield video_capture
    finally:
        video_capture.release()       

def read_stream_until_quit(stream):
    while True:
        ret, frame = stream.read()
        yield frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
# todo freezes if negative locations -> should be no negative locations
def init_face_detector_cnn():
    detector = dlib.cnn_face_detection_model_v1("face-detection-benchmark/models/face_detection/mmod_human_face_detector.dat")
    scale = FACE_DETECTION_INPUT_WIDTH / DISPLAY_WIDTH

    def unscale(result):
        return dlib.rectangle(int(result.left()*1.0/scale), int(result.top()*1.0/scale),
                              int(result.right()*1.0/scale), int(result.bottom()*1.0/scale))

    def run(frame):
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        return [unscale(r.rect) for r in detector(small_frame, 0)]  # todo check confidence

    return run

def init_face_cropper():
    landmark_model = dlib.shape_predictor("face-detection-benchmark/models/face_cropping/shape_predictor_5_face_landmarks.dat")

    def run(frame, face):
        shape = landmark_model(frame[:, :, ::-1], face) 
        return dlib.get_face_chip(frame, shape)[:, :, ::-1]

    return run


def init_feature_extractor():
    feature_model = cv2.dnn.readNetFromTorch("face-detection-benchmark/models/feature_extraction/nn4.small2.v1.t7")

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
                # print(frame.shape)

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


main()

