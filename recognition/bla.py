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
        return dlib.get_face_chip(frame, shape)[:, :, ::-1]  # todo 96

    return run


def init_feature_extractor():
    feature_model = cv2.dnn.readNetFromTorch("models/feature_extraction/nn4.small2.v1.t7")

    def run(face):
        blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), swapRB=False,
                                     crop=False)  # todo resize should not be necessary
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
        distances = [np.linalg.norm(face - saved_face) for saved_face in face_database]
        print(distances)
        matches = [d for d in distances if d < 1.0]
        return len(matches) > 0

    return run
