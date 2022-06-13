# FaceReader.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Mon. 06/13/2022
# Trevor Reynen

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import dlib         # dlib is a toolkit for making machine learning and data analysis applications.
import numpy as np  # NumPy is an important library used for numerical computing.
import sys          # sys is used to manipulate different parts of the Python runtime environment.
import os           # os provides functions for interacting with the operating system.

# Read face, analyze how many times you yawn.
# As soon as we detect the large separation between our upper and lower lips in our mouth
# landmarks, we count once for yawning

PREDICTOR_PATH = './images/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
print(type(detector))
print(type(predictor))


def get_landmarks(image):
    rects = detector(image, 1)

    # More than one face detected.
    if len(rects) > 1:
        return 'error'

    # No face detected.
    if len(rects) == 0:
        return 'error'

    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def annotate_landmarks(image, landmarks):
    image = image.copy()

    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])

        cv2.putText(image, str(idx), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (0, 0, 255))

        cv2.circle(image, pos, 3, (0, 255, 255))

    return image


# Get y_coordinate of  top_lip from our landmarks.
def top_lip(landmarks):
    top_lip_pts = []

    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])

    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])

    # Shape of top_lip_pts (6, 1, 2).
    #top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))

    # Shape of top_lip_mean (1, 2)
    # [[171.         236.66666667]]
    top_lip_mean = np.mean(top_lip_pts, axis=0)

    # Return the average of y coordinates of upper lip.
    return int(top_lip_mean[:, 1])


# Get y_coordinate of lower lip.
def bottom_lip(landmarks):
    bottom_lip_pts = []

    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])

    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])

    # top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)

    # Return the average of y coordinates of bottom lip.
    return int(bottom_lip_mean[:, 1])


# Return image_with_landmarks and distance between top_lip and bottom_lip.
def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == 'error':
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)

    return image_with_landmarks, lip_distance


yawns = 0
yawn_status = False
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    image_landmarks, lip_distance = mouth_open(frame)

    prev_yawn_status = yawn_status

    # Revise. This should be proportion of head frame.
    # If the distance is greater than 5% of vertical distance of head frame, it is a yawn.
    #if lip_distance > 10:
    if lip_distance > 20:
        yawn_status = True

        cv2.putText(frame, 'Subject is Yawning', (50,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        output_text = ' Yawn Count: ' + str(yawns + 1)

        cv2.putText(frame, output_text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
    else:
        yawn_status = False

    # We do not double count or count it wrong.
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    # Three second delay dlib processing to the background.
    cv2.imshow('Live Landmarks', image_landmarks)
    cv2.imshow('Face Reader', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

cap.release()
cv2.destroyAllWindows()

