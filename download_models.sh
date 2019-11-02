DIRECTORY="models"

if [ -d "$DIRECTORY" ]; then
    echo "Directory \"$DIRECTORY\" already exists, please delete and run this script again"
    exit 1
fi

# create output dirs
mkdir "$DIRECTORY"
mkdir "$DIRECTORY/face_detection"
mkdir "$DIRECTORY/landmark_detection"
mkdir "$DIRECTORY/feature_extraction"

# face detection models
cd "$DIRECTORY/face_detection"

wget  http://dlib.net/files/mmod_human_face_detector.dat.bz2
bunzip2 mmod_human_face_detector.dat.bz2

wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
	
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt

cd ../..

# face cropping / landmark detection models

cd "$DIRECTORY/landmark_detection"

wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
bunzip2 shape_predictor_5_face_landmarks.dat.bz2

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

cd ../..

# face feature extraction

cd "$DIRECTORY/feature_extraction"

wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2

wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7