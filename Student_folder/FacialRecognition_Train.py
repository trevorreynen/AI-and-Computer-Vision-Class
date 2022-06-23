# FacialRecognition_Train.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Wed. 06/15/2022
# Trevor Reynen

# Train Model


# Imports.
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


# Get the training data we previously made.
data_path = './assets/images/faces/user/'

# Store the list of file names in the directory.
onlyFiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Extract faces from webcam view.
def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    #print(faces)

    if faces is ():
        return img, []

    # Assume we only have single face.
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

# Test our facial recognition classifier.
face_classifier = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_frontalface_default.xml')


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

Training_Data, Labels = [], []

# Open training image in our data path. Build a numpy array for training data.
for i, files in enumerate(onlyFiles):
    image_path = data_path + onlyFiles[i]

    # Reading each image in the path.
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Storing it into numpy array.
    Training_Data.append(np.asarray(images, dtype=np.uint8))

    Labels.append(i)


# Create a numpy array for both training data and labels.
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer model.
# NOTE: For OpenCV 3.0 use: 'model = cv2.face.createLBPHFaceRecognizer()'
model = cv2.face.LBPHFaceRecognizer_create()

# Train the models by feeding our training data and labels.
model.train(np.asarray(Training_Data), np.asarray(Labels))
print('Model Trained Successfully.')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to its prediction model. 'results' comprises of a tuple containing the label
        # and the confidence value. Label is which face of yourself in the dataset.
        results = model.predict(face)
        print('Prediction Confidence', results)

        # Output value go up to 500. Lower value the better.
        if results[1] < 500:
            # Calculate confidence value which is a percentage.
            confidence = int(100 * (1 - (results[1]) / 400))

            display_string = str(confidence) + '% Confidence it\'s a user'

        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 210, 150), 2)

        #image = cv2.flip(image, 1)

        # Decide whether the face is your face or not. 85 confidence works well.
        if confidence > 85:
            cv2.putText(image, 'Unlock', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Recognition', image)
        else:
            cv2.putText(image, 'Locked', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)
    except:
        # Any error in the try block will try this code.
        cv2.putText(image, 'No Face Found', (220, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, 'Locked', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Recognition', image)
        pass

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

cap.release()
cv2.destroyAllWindows()

