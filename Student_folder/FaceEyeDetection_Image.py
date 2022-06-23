# FaceEyeDetection_Image.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Wed. 06/08/2022
# Trevor Reynen

# Face & Eye Detection using HAAR Cascade classifiers.


# Imports.
import cv2


# Search classifier opencv/data/haarcascades/.

# Cascade Classifiers Flow.
# 1. Load Classifier.
# 2. Pass Image to Classifier/Detector.
# 3. Get Location/ROI (Region of Interest) for Detected Objects.
# 4. Draw rectangle over Detected Objects.

# We use some pre-trained classifiers that have been provided by OpenCV.
# These pre-trained classifiers are stored as .xml files.

# Create face_classifier object.
face_classifier = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_frontalface_default.xml')

# Create eye_classifier object.
eye_classifier = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_eye.xml')

# Load our image and convert it into grayscale.
image = cv2.imread('./assets/images/ManyFaces.jpg')
#image = cv2.imread('./assets/images/ManyFaces2.jpg')
#image = cv2.imread('./assets/images/ManyFaces3.jpg')
#image = cv2.imread('./assets/images/candy.jpg')

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a 4-element tuple.
# If faces are found, it returns array of positions of detected faces as Rect(x, y, w, h).
faces = face_classifier.detectMultiScale(gray_img, 1.3, 5)
print('List for location for faces:')
print(faces)
print()

if faces == ():
    print('No faces found.')

# Shows the original imported image before detections.
#cv2.imshow('Original', image)

# We iterate through our faces array and draw a rectangle over each face in faces.
# x, y => upper-left corner coordinates of face.
# width(w) of rectangle in the face.
# height(h) of rectangle in the face.
# grey means the input image to the detector.
for (x, y, w, h) in faces:
    # Draw pink rectangle around face.
    cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(1000)

    # Crop the face out of full frame.
    roi_gray = gray_img[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)

    print('List for location for eyes')
    print(eyes)

    for (ex, ey, ew, eh) in eyes:
        # Draw the rectangle around eyes.
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
        cv2.imshow('Eye Detection', image)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

