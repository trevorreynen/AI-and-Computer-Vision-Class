# ObjectTracking_Ex3.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Mon. 06/20/2022
# Trevor Reynen

# Tracking object with the specific color.


# Imports.
import cv2
import numpy as np


# Initialize Camera.
cap = cv2.VideoCapture(0)

# Define range of color for our filter in HSV.
lower_purple = np.array([130, 50, 90])
upper_purple = np.array([170, 255, 255])

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([120, 255, 255])

# Create empty points array.
points = []

# Get default camera window size.
ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 0

while True:
    ret, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only color we want.
    #mask = cv2.inRange(hsv_img, lower_purple, upper_purple)
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # It outputs the contours and hierarchy.
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # This is the extra code to draw the contours.
    #cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    #frame = cv2.flip(frame, 1)
    #cv2.imshow('Contours', frame)

    # Create empty center array to store centroid center of mass.
    center = int(Height / 2), int(Width / 2)

    if len(contours) > 0:
        # Get the largest contour and it's center.
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)

        # From this moment, you can extract useful data like area, centroid, and center of weight.
        try:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        except:
            center = int(Height / 2), int(Width / 2)

        # Allow only contours that have a larger than 25 pixel radius.
        if radius > 25:
            # Draw the circle around the filtered object.
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            # Draw the center of weight.
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

    # Log center points.
    points.append(center)

    if radius > 25:
        # Loop over the set of tracked points.
        for i in range(1, len(points)):
            try:
                # This is the code to draw the line.
                cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
            except:
                pass

        # Make frame count zero.
        frame_count = 0
    else:
        # Count frames.
        frame_count += 1

        # If we count 10 frames without object, lets delete our trail.
        if frame_count == 10:
            points = []
            frame_count = 0

    #frame = cv2.flip(frame, 1)

    # Display our object tracker.
    cv2.imshow('Object Tracker', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

# Release camera and close any open windows.
cap.release()
cv2.destroyAllWindows()

