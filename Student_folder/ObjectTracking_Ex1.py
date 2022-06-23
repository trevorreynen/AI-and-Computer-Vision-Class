# ObjectTracking_Ex1.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Thu. 06/16/2022
# Trevor Reynen

# Here, we create a simple application which tracks some points in a video.
# Lucas-Kanade Optical Flow in OpenCV.


# Imports.
import cv2
import numpy as np


# OpenCV provides all these in a single function, cv.calcOpticalFlowPyrLK(). Check the slides for
# details of parameter.

# Load video.
cap = cv2.VideoCapture('./assets/images/walking.avi')
#cap = cv2.VideoCapture('./assets/images/cars.avi')

# OpenCV has a function, cv2.goodFeaturesToTrack().
# It finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if
# you specify it).

# Set parameters for ShiTomasi corner detection.
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Set parameters for Lucas-Kanade Optical Flow.
lucas_kanade_params = dict(winSize=(15, 15),
                           maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# Create some random colors. Used to create our trails for object movement in the image. Last
# parameter is the dimension of matrix.
color = np.random.randint(0, 255, (100, 3))
print(color)

# Take first frame and find corners in it.
# OpenCV has a function, cv2.goodFeaturesToTrack().
# It finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if
# you specify it).
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Find initial corner locations.
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes. Mask is zero-like matrix.
mask = np.zeros_like(prev_frame)

# After finding initial points, we iteratively track those points.
while 1:
    ret, frame = cap.read()

    cv2.imshow('Original', frame)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow. For the function cv.calcOpticalFlowPyrLK() we pass the previous
    # frame, previous points and next frame. It returns next points along with some status numbers
    # which has a value of 1 if next point is found, else zero.

    # We iteratively pass these next points as previous points in next step.
    new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                           frame_gray,
                                                           prev_corners,
                                                           None,
                                                           **lucas_kanade_params)

    # Select and store good points.
    good_new = new_corners[status == 1]
    good_old = prev_corners[status == 1]
    #print(good_new)

    # Draw the tracks.
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # a, b is the coordinate of next position.
        a, b = new.ravel()
        # c, d is the coordinate of previous position.
        c, d = old.ravel()

        # Draw the line between two points between 2 frames.
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)

        # Draw the circle around the object. frame is the original image.
        newframe = cv2.circle(frame, (a, b), 6, color[i].tolist(), -1)

    # Tracking line.
    #edited_img = mask

    # We can draw the circle first.
    #edited_img = newframe

    # Image Addition (tracking line and draw the circle). Calculates the per-element sum of two
    # arrays or an array and a scalar.
    edited_img = cv2.add(newframe, mask)

    # Show Optical Flow
    cv2.imshow('Optical Flow - Lucas-Kanade', edited_img)

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

    # Now update the previous frame and previous points.
    prev_gray = frame_gray.copy()
    prev_corners = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()

