# FaceLandMarkImage.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Mon. 06/13/2022
# Trevor Reynen

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import dlib         # dlib is a toolkit for making machine learning and data analysis applications.
import numpy as np  # NumPy is an important library used for numerical computing.
import sys          # sys is used to manipulate different parts of the Python runtime environment.
import os           # os provides functions for interacting with the operating system.

# Finding facial landmarks using dlib.
# This program only works for a single face.

# Facial landmarks give you 68 key points.
# Facial Landmarks Number Order.
# MOUTH_POINTS = 48 to 61
# RIGHT_BROW_POINTS = 17 to 21
# LEFT_BROW_POINTS = 22 to 27
# RIGHT_EYE_POINTS = 36 to 42
# LEFT_EYE_POINTS = 42 to 48
# NOSE_POINTS = 27 to 35
# JAW_POINTS = 0 to 17

# Name of our model file.
PREDICTOR_PATH = './images/shape_predictor_68_face_landmarks.dat'

# Load the file to create predictor objects for generating landmarks.
# Shape-predictor takes the path to dlib's pre-trained facial landmark detector.
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Create face detector object. Returns the default face detector.
detector = dlib.get_frontal_face_detector()
print(type(detector))
print(type(predictor))


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


# Return facial landmarks given the image.
def get_landmarks(image):
    # Ask the detector to find the bounding boxes of each face.
    # rects is an array of bounding boxes of faces detected.
    rects = detector(image, 1)
    #print('Value of face detector in get_landmarks')
    # rectangles[[(141, 201) (409, 468)]]
    #print(rects)

    if len(rects) > 1:
        raise TooManyFaces

    if len(rects) == 0:
        raise NoFaces

    #print('predictor in get_landmarks')

    # Get the landmarks/parts for the face in box d.
    #print(predictor(image, rects[0]).parts())

    # rects[0] provide the bounding box for first face.
    # Reformatting the resulting array into numpy array.
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


# This function will plot the number and circle of key feature onto the face.
def annotate_landmarks(image, landmarks):
    image = image.copy()

    # Enumerate returns an iterator yielding pairs of array coordinates and values.
    for idx, point in enumerate(landmarks):
        # Plot landmarks on the face.
        # Point is list (2d matrix)
        #print(point)
        pos = (point[0, 0], point[0, 1])

        # Plot the number for each landmarks onto the face.
        cv2.putText(image, str(idx), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0, 0, 255))

        # Put circle around those points.
        cv2.circle(image, pos, 6, (0, 255, 255))

    return image


image = cv2.imread('./images/single-face1.jpg')
#image = cv2.imread('./images/single-face2.jpg')
#image = cv2.imread('./images/Trump.jpg')

cv2.imshow('Original', image)

landmarks = get_landmarks(image)
print('Landmarks')
print(landmarks)
print()
print('Shape of landmarks', landmarks.shape)

image_with_landmarks = annotate_landmarks(image, landmarks)

cv2.imshow('Result', image_with_landmarks)

# Takes the file name including extension, separates name and extension, then numbers the file name
# to prevent files from being replaced.
def uniqueFile(file):
    fName, fExt = os.path.splitext(file)
    i = 1
    saveFile = fName + '-' + str(i) + fExt

    while os.path.exists(saveFile):
        saveFile = fName + '-' + str(i) + fExt
        i += 1

    return saveFile


# Store the resulting image into the hard drive.
cv2.imwrite(uniqueFile('./images/saved/FaceLandMarkImage.jpg'), image_with_landmarks)
cv2.waitKey(20000)

cv2.destroyAllWindows()

