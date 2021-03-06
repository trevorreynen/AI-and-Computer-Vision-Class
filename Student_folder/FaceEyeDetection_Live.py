# FaceEyeDetection_Live.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Wed. 06/08/2022
# Trevor Reynen

# Let's make a live face and eye detection, keeping the face in view at all times.


# Imports.
import cv2


# Create face_classifier object.
face_classifier = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_frontalface_default.xml')

# Create eye_classifier object.
eye_classifier = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_eye.xml')


# The function will detect both the faces and eyes given in the image.
def face_eye_detector(image, size=0.3):
    # Convert image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Our classifier returns the ROI of the detected face as a 4-element tuple.
    # If faces are found, it returns array of positions of detected faces as Rect(x, y, w, h).
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # When no faces are detected, face_classifier returns empty tuple.
    if faces == ():
        return image

    # We iterate through our faces array and draw a rectangle over each face in faces.
    for (x, y, w, h) in faces:
        # Make sure our rectangle is bigger than face.
        x = x - 25
        w = w + 25
        y = y - 25
        h = h + 25

        # Draw rectangle around face.
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # We crop the face out of image.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Once we get these locations, we can create ROI for the face and apply detection on this ROI.
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    #roi_color = cv2.flip(roi_color, 1)

    # Show cropped image.
    #return (roi_color)

    # Show full image.
    return image


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('Live Face and Eye Extractor', face_eye_detector(frame))

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

cap.release()
cv2.destroyAllWindows()

