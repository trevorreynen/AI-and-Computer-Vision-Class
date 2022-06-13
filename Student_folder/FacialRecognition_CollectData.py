# FacialRecognition_CollectData.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/14/2022
# Trevor Reynen

# This program will collect your face for training.
# Need to create './faces/user/' directory (just the folders).

# Build the app to recognize your face only.

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import dlib         # dlib is a toolkit for making machine learning and data analysis applications.
import numpy as np  # NumPy is an important library used for numerical computing.
import sys          # sys is used to manipulate different parts of the Python runtime environment.
import os           # os provides functions for interacting with the operating system.

# Load HAAR face classifier.
face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')


# This function will return our cropped face.
def face_extractor(img):
    # Function detects faces and returns the cropped face. If no face detected, it returns none.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found.
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


count = 0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if face_extractor(frame) is not None:
        count += 1

        # Resize our face to 200x200.
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name.
        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count.
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        # If face is not found, do nothing.
        print('Face not found.')
        pass

    if cv2.waitKey(1) == 13 or count == 300:  # 13 is the Enter key.
        break


cap.release()
cv2.destroyAllWindows()

print('Collecting Samples Complete')

