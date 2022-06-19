# ObjectTracking_Ex2.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Thu. 06/16/2022
# Trevor Reynen

# Dense Optical Flow in OpenCV.


# Imports.
import cv2
import numpy as np


# Load video.
cap = cv2.VideoCapture('./images/walking.avi')
# cap = cv2.VideoCapture('./images/cars.avi')

# Get first frame.
ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# HSV is new way of representing color (Hue, Saturation, and Value)
# Make dimensional 255
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255
print(hsv)

while cap.isOpened():
    ret, frame2 = cap.read()

    height, width = frame2.shape[:2]

    cv2.imshow('Original', frame2)

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculates an Optical Flow.
    flow = cv2.calcOpticalFlowFarneback(previous_gray, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Use flow to calculate the magnitude (speed) and angle of motion.
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Colors are used to reflect movement with Hue being direction and Value (brightness/intensity)
    # being speed.
    hsv[..., 0] = angle * (180 / (np.pi / 2))
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    final_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Show out demo of Dense Optical Flow.
    cv2.imshow('Dense Optical Flow', final_img)

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

    # Store current image as previous image.
    previous_gray = next

cap.release()
cv2.destroyAllWindows()

