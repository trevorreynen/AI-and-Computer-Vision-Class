# BlurringImage.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Mon. 06/06/2022
# Trevor Reynen


# Imports.
import cv2
import numpy as np


# image = cv2.imread('./images/scene1.jpg')
image = cv2.imread('./assets/images/single-face1.jpg')

a = 1

while True:
    cv2.imshow('Original Image', image)

    # Blurry image using convolution with kernel.
    # We will discuss convolution and kernel in a later chapter.

    # Create 7 x 7 kernel (matrix with 7 rows and 7 columns).
    # We multiple by 1 / 49 to normalize the matrix.
    kernel_7x7 = np.ones((7, 7), np.float32) / 49

    # Simple minded way to prevent the print below from spamming console.
    if a < 2:
        print(kernel_7x7)
        a = 2

    blurred = cv2.filter2D(image, -1, kernel_7x7)
    cv2.imshow('7 x 7 Kernel Blurring', blurred)

    # Convolution of image with Gaussian kernel.
    Gaussian = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow('Gaussian Blurring', Gaussian)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cv2.destroyAllWindows()

