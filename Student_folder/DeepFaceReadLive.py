# DeepFaceReadLive.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Mon. 06/27/2022
# Trevor Reynen

# DeepFaceReader combines emotion, age, and gender recognition.
# At first we will detect face, then find out emotion (facial expression), estimated age of person
# and predicted gender.


# Imports.
import cv2, dlib, keras, sys
import numpy as np
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from pathlib import Path
from wide_resnet import WideResNet


print('Path for Python:', sys.executable, '\n')
print(cv2.__version__, ', Path:', cv2.__path__, '\n')
print(dlib.__version__, ', Path:', dlib.__path__, '\n')
print(np.__version__, ', Path:', np.__path__, '\n')
print(keras.__version__, ', Path:', keras.__path__, '\n')

# This is the classifier for emotion detection using little VGG model. This model 60% accuracy.
classifier = load_model('./assets/models/trained/emotion_little_vgg_3.h5')

# Load our Haarcascade classifier for detecting face.
face_classifier = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_frontalface_default.xml')

# Load our pretrained model for Gender and Age detection.
pretrained_model = 'https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5'

print(pretrained_model)

# We need specify modhash when we load the model.
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

# We have 6 facial expression.
emotion_classes = { 0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise' }

print('Type of emotion')
print(emotion_classes)


# Face Detector function.
def face_detector(img):
    # Convert image to grayscale for faster detection.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Array of face locations.
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return False, (0, 0, 0, 0), np.zeros((1, 48, 48, 3), np.uint8), img

    # Images of only faces (roi).
    allfaces = []

    # Locations of faces.
    rects = []

    # x, y, w, h is location of face.
    for (x, y, w, h) in faces:
        # Put rectangle around the image.
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop face region.
        roi = img[y:y+h, x:x+w]
        allfaces.append(roi)
        rects.append((x, w, y, h))

    return True, rects, allfaces, img


# Define our model parameters for age and gender detector.
depth = 16
k = 8
weight_file = None
margin = 0.4
image_dir = None

# Get weight_file.
if not weight_file:
    weight_file = get_file(fname='weights.28-3.73.hdf5',
                           origin=pretrained_model,
                           file_hash=modhash,
                           cache_subdir='assets\\models\\pretrained\\',
                           cache_dir=Path(sys.argv[0]).resolve().parent)

# Load model and weights for age and gender detection which is WideResNet.
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)
print(model)


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # rects is list of rects for face region.
    # faces is list of regions of interest (ROI) for face crop (we may detect multiple faces).
    # image is the image file.
    ret, rects, faces, image = face_detector(frame)

    preprocessed_faces_age = []
    preprocessed_faces_emo = []

    # If we detect face, ret is true.
    if ret:
        # This works for multiple faces.
        for (i, face) in enumerate(faces):
            # Preprocess the image for age detector.
            face_age = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
            preprocessed_faces_age.append(face_age)

            # Preprocess the image for emotion detector. Convert image to gray.
            face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Resize the image.
            face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation=cv2.INTER_AREA)

            # Normalize the image.
            face_gray_emo = face_gray_emo.astype('float') / 255.0

            # Convert it into numpy array.
            face_gray_emo = img_to_array(face_gray_emo)

            # Add one more dimension.
            face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
            preprocessed_faces_emo.append(face_gray_emo)

        # Make a prediction for Age and Gender.
        results = model.predict(np.array(preprocessed_faces_age))
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)

        # Predicted ages in the list containing the predicted age for each face.
        predicted_ages = results[1].dot(ages).flatten()

        # Go through each face. Make a prediction for Emotion.
        emo_labels = []

        for (i, face) in enumerate(faces):
            preds = classifier.predict(preprocessed_faces_emo[i])[0]

            # Append the emotional label for each face.
            emo_labels.append(emotion_classes[preds.argmax()])

        # Overlay detected emotion on our faces.
        for (i, face) in enumerate(faces):
            # Draw results for Age and Gender.
            label = '{}, {}, {}'.format(int(predicted_ages[i]),
                                        'F' if predicted_genders[i][0] > 0.6 else 'M',
                                        emo_labels[i])

            # Make sure the label is in correct position above face.
            label_position = (rects[i][0] + int(rects[i][1] / 2), abs(rects[i][2] - 10))
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', image)

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

cap.release()
cv2.destroyAllWindows()

