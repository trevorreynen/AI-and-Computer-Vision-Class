# PedestrianDetection.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Thu. 06/09/2022
# Trevor Reynen

# Locate pedestrians in an video/image.

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import numpy as np  # NumPy is an important library used for numerical computing.

# Create body_classifier to detect pedestrians.
body_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_fullbody.xml')

# Initiate video capture for video file.
cap = cv2.VideoCapture('./images/walking.avi')
#cap = cv2.VideoCapture('./images/P1033741.mp4')

# Loop once video has successfully loaded.
while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body.
    # Larger scaling factor makes detection faster. Larger min-neighbor makes it more accurate.
    bodies = body_classifier.detectMultiScale(gray, 1.2, 2)
    #print('Body List')
    #print(bodies)

    # Get video width and height for placement of text.
    vidWidth = cap.get(3)
    vidHeight = cap.get(4)
    xPlacement = vidWidth * 0.01
    yPlacement = vidHeight * 0.95
    org = (int(xPlacement), int(yPlacement))

    # Extract bounding boxes for any bodies identified.
    for (x, y, w, h) in bodies:
        # Add a box around each pedestrian detected.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow('Pedestrians', frame)

        # Add the text which updates live based on number of cars actively in frame.
        label = 'Pedestrians in frame: ' + str(len(bodies))
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cap.release()
cv2.destroyAllWindows()

