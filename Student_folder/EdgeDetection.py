# EdgeDetection.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Lab Due: Tue. 06/07/2022
# Trevor Reynen

# Demonstrates how to use .Canny() for edge detection.

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import numpy as np  # NumPy is an important library used for numerical computing.

#image = cv2.imread('images/input.jpg', 0)
image = cv2.imread('images/ManyFaces.jpg', 0)
#image = cv2.imread('images/elephant.jpg', 0)

while True:
    cv2.imshow('Original Image', image)

    # Use Canny for edge detection.
    canny = cv2.Canny(image, 70, 100)

    cv2.imshow('Canny', canny)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cv2.destroyAllWindows()

