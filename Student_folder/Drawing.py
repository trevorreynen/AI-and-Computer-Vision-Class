# Drawing.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/07/2022
# Trevor Reynen

# Draw the picture using pencil effects.


# Imports.
import cv2


# Initialize camera.
cap = cv2.VideoCapture(0)

# This is the function to process the image.
def sketch(image):
    # Convert image to grayscale. Gaussian Blur needs a gray image.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth image by removing the noise using Gaussian Blur.
    img_gray_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # Extract edges using Canny.
    canny_edges = cv2.Canny(img_gray_blur, 5, 100)

    # Invert black to white and white to black.
    ret, mask = cv2.threshold(canny_edges, 10, 255, cv2.THRESH_BINARY_INV)

    # Produce mirror image.
    mask = cv2.flip(mask, 1)

    #return image
    #return image_gray
    #return img_gray_blur
    return canny_edges
    #return mask


# Use while loop to continuously pull the image from webcam using cap.read().
while True:
    # Ret tells you whether you are successful. Frame is your live image.
    ret, frame = cap.read()

    cv2.imshow('Original Image', frame)
    cv2.imshow('Live Drawing', sketch(frame))

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

cap.release()
cv2.destroyAllWindows()

