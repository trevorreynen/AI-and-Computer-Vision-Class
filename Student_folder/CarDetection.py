# CarDetection.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Thu. 06/09/2022
# Trevor Reynen

# Detect cars from Cars.avi in Clips folder.
# Draw a rectangle around each car and show how many cars are actively in each frame.

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import numpy as np  # NumPy is an important library used for numerical computing.
from imutils.object_detection import non_max_suppression

# Create our body classifier, car_slassifier.
car_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_car.xml')

# Initiate video capture for video file.
cap = cv2.VideoCapture('./images/cars.avi')

# Loop once video is successfully loaded.
while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_classifier.detectMultiScale(gray, 1.1, 1)

    # Get video width and height for placement of text.
    vidWidth = cap.get(3)
    vidHeight = cap.get(4)
    xPlacement = vidWidth * 0.01
    yPlacement = vidHeight * 0.1
    org = (int(xPlacement), int(yPlacement))

    for (x, y, w, h) in cars:
        # Add a box around each car detected.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Cars', frame)

        # Add the text which updates live based on number of cars actively in frame.
        pick = non_max_suppression(cars, probs=None, overlapThresh=0.65)
        label = 'Cars in frame: '
        numCars = len(pick)
        combined = label + str(numCars)
        cv2.putText(frame, str(combined), org, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cap.release()
cv2.destroyAllWindows()

