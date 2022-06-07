# FaceEyeDetectionImage.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Lab Due: Wed. 06/08/2022
# Trevor Reynen

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import numpy as np  # NumPy is an important library used for numerical computing.

# Face & Eye Detection using HAAR Cascade classifiers.

# Search classifier opencv/data/haarcascades/.

# Casscade Classifiers Flow.
# 1. Load Classifier.
# 2. Pass Image to Classifier/Detector.
# 3. Get Location/ROI (Region of Interest) for Detected Objects.
# 4. Draw rectangle over Detected Objects.

# We use some pre-trained classifiers that have been provided by OpenCV.
# These pre-trained classifiers are stored as .xml files.


# Create face_classifier object.
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Create eye_classifier object.
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')


# Load our image and convert it into grayscale.
img = cv2.imread('images/ManyFaces.jpg')
# img = cv2.imread('images/ManyFaces2.jpg')
# img = cv2.imread('images/ManyFaces3.jpg')
# img = cv2.imread('images/candy.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a 4-element tuple.
# If faces are found, it returns array of positions of detected faces as Rect(x, y, w, h).
# It is the list of list (locations of many faces)
faces = face_classifier.detectMultiScale(gray_img, 1.3, 5)
print('List for location for faces:')
print(faces)
print('\n')


if faces == ():
	print('No faces found.')


a = 1
while True:
    # Shows the original importanted image before detections.
    #cv2.imshow('Original', img)


    # We iterate through our faces array and draw a rectangle over each face in faces.
    # x, y => upperleft corner coordinates of face.
    # width(w) of rectangle in the face.
    # height(h) of rectangle in the face.
    # grey means the input image to the detector.
    for (x, y, w, h) in faces:
        # Draw pink rectangle around face.
        cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
        cv2.imshow('Face Detection', img)

        # We crop the face out of image.
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        if a < 2:
            print('List for location for eyes')
            print(eyes)
            print('\n')
            a = 2

        for (ex, ey, ew, eh) in eyes:
            # Draw the rectangle around eyes.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            cv2.imshow('Eye Detection', img)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cv2.destroyAllWindows()

