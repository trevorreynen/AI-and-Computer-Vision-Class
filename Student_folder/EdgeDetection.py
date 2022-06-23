# EdgeDetection.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/07/2022
# Trevor Reynen

# Demonstrates how to use .Canny() for edge detection.


# Imports.
import cv2


#image = cv2.imread('./assets/images/input.jpg', 0)
image = cv2.imread('./assets/images/ManyFaces.jpg', 0)
#image = cv2.imread('./assets/images/elephant.jpg', 0)

while True:
    cv2.imshow('Original Image', image)

    # Use Canny for edge detection.
    canny = cv2.Canny(image, 70, 100)

    cv2.imshow('Canny', canny)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cv2.destroyAllWindows()

