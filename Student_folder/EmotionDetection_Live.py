# EmotionDetection_Live.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Thu. 06/23/2022
# Trevor Reynen

# We will use deep learning models built in previous lectures to detect live emotion.
# You need ./assets/models/trained/emotion_little_vgg_3.h5, which is produced by
# EmotionDetectionTrain.py


# Imports.
import cv2
import numpy as np

# Keras Imports.
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array


# Loading our saved model.
classifier = load_model('./assets/models/trained/emotion_little_vgg_3.h5')

# Haarcascades is the type of object detector ( we will use face detection capability).
face_classifier = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_frontalface_default.xml')

# This func returns rectangles of face locations, the cropped image of face, and complete image.
def face_detector(img):
    # Convert image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Return array of locations of faces.
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # If no faces, return some default value.
    if faces is ():
        return ((0, 0, 0, 0), np.zeros((48, 48), np.uint8), img)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face.
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the image to face out of the image.
        # We need tightly cropped face.
        roi_gray = gray[y:y+h, x:x+w]

    try:
        # Resize roi_gray to the image which classifier accepts.
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return ((x, w, y, h), np.zeros((48, 48), np.uint8), img)

    return ((x, w, y, h), roi_gray, img)


# Get our class labels.
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_data_dir = './assets/fer2013/validation'

num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 16

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_rows, img_cols),
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              batch_size=batch_size,
                                                              shuffle=False)

class_labels = validation_generator.class_indices
class_labels = { v: k for k, v in class_labels.items() }
classes = list(class_labels.values())
print(class_labels)


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect the face.
    rect, face, image = face_detector(frame)

    if np.sum([face]) != 0.0:
        # Normalize the face data.
        roi = face.astype('float') / 255.0

        # Converts a PIL Image instance to a Numpy array.
        roi = img_to_array(roi)

        # Reshape the matrix so that face data can be used for our deep learning model.
        roi = np.expand_dims(roi, axis=0)

        # Make a prediction on the ROI (face).
        preds = classifier.predict(roi)[0]

        # Lookup the emotion class.
        label = class_labels[preds.argmax()]
        label_position = (rect[0] + int((rect[1] / 2)), rect[2] + 25)

        # Put label of prediction near the face.
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('All', image)

    if cv2.waitKey(1) == 13:  # 13 is the Enter key.
        break

cap.release()
cv2.destroyAllWindows()

