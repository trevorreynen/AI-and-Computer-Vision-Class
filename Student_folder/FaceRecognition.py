# FaceRecognition.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Wed. 06/29/2022
# Trevor Reynen

# NOTE: The model needed for the labs FaceRecognition.py and FaceRecognition_Matching.py, called
# "vgg_face_weights.h5" is over 100 MB. For now, I will not be uploading it to GitHub inside this
# repository. If I find a link to the model I will either add it here or add code to the two
# FaceRecognition programs to download the model if it isn't already.

# In this section, we use the VGGFace model to identify yourself. It uses one-shot learning, so we
# only need one picture for our recognition system.

# The VGG-Face CNN descriptors are computed using our CNN implementation based on the
# VGG-Very-Deep-16 CNN architecture and are evaluated on the labeled faces in the Wild and the
# YouTube Faces Dataset.

# Place photos of people (one face visible) in a folder called
# './assets/images/face_recognition/person/' put our picture into person folder for testing on a
# webcam. Faces are extracted using the haarcascade_frontface_default detector model. Extracted
# faces are placed in the folder called './assets/images/face_recognition/group_of_faces'.

# We extract the faces needed for our one-shot learning model, it will load 5 extracted faces.


# Imports.
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Keras Imports.
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Convolution2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img


# Loading our HAARCascade face detector.
face_detector = cv2.CascadeClassifier('./assets/Haarcascades/haarcascade_frontalface_default.xml')

# Directory of images with people to extract faces from.
mypath = './assets/images/face_recognition/person/'

image_file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print('Collected image names:', image_file_names)

for image_name in image_file_names:
    person_image = cv2.imread(mypath + image_name)
    face_info = face_detector.detectMultiScale(person_image, 1.3, 5)

    for (x, y, w, h) in face_info:
        face = person_image[y:y + h, x:x + w]
        roi = cv2.resize(face, (128, 128), interpolation=cv2.INTER_CUBIC)

    path = './assets/images/face_recognition/group_of_faces/' + 'face_' + image_name
    cv2.imwrite(path, roi)
    cv2.imshow('face', roi)
    cv2.waitKey(2000)


cv2.destroyAllWindows()


# Loads image from path and resizes it. Notice that VGG model expects 224x224x3 sized input images.
# Here, 3rd dimension refers to number of channels or RGB colors.

# Explanation of this code:
# Keras works with batches of images. So, the first dimension is used for the number of samples (or
# images) you have. When you load a single image, you get the shape of one image, which is (size1,
# size2, channels). In order to create a batch of images, you need an additional dimension:
# (samples, size1, size2, channels). The preprocess_input function is meant to adequate your image
# to the format the model requires. Some models use images with values ranging from 0 to 1. Others
# from -1 to +1. Others use the 'caffe' style, that is not normalized, but is centered.
def preprocess_image(image_path):
    # This PIL image instance.
    img = load_img(image_path, target_size=(224, 224))

    # Convert to numpy array.
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


# Loads the VGGFace model.
def loadVggFaceModel():
    model = Sequential()
    # First block. Adding zero on top, bottom, left and right.
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    # 64 3x3 filters with relu as the activation. For padding, I think the default padding is 1
    # (valid padding). , and the default stride is 1.
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Second bock.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Third block.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fourth block.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fifth block.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    # If you don't specify anything, no activation is applied (ie. 'linear' activation: a(x) = x).
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    model.load_weights('./assets/models/vgg_face_weights.h5')
    print(model.summary())

    # We'll use previous layer of the output layer for representation.
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor


model = loadVggFaceModel()
print('Model Loaded')


# Vector Similarity.
# We’ve represented input images as vectors. We will decide both pictures are same person or not
# based on comparing these vector representations. Now, we need to find the distance of these
# vectors. There are two common ways to find the distance of two vectors: cosine distance and
# euclidean distance. Cosine distance is equal to 1 minus cosine similarity. No matter which
# measurement we adapt, they all serve for finding similarities between vectors.
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# Test model using your webcam. This code looks up the faces you extracted in the 'group_of_faces'
# and uses the similarity (Cosine similarity) to detect which faces is most similar to the one
# being extracted with your webcam.

# Points to your extracted faces.
people_pictures = './assets/images/face_recognition/group_of_faces/'

# This is dictionary with its key as person name, value as vector representation.
all_people_faces = dict()

for file in listdir(people_pictures):
    person_face, extension = file.split('.')
    img = preprocess_image('./assets/images/face_recognition/group_of_faces/%s.jpg' % (person_face))

    # We can represent images 2622 dimensional vector.
    face_vector = model.predict(img)[0, :]
    all_people_faces[person_face] = face_vector
    print(person_face, face_vector)

print('Face representations retrived successfully.')


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# This code looks up the faces you extracted in the 'group_of_faces' and uses the similarity
# function to detect which faces is most similar to the one being extracted with your webcam.

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    faces = face_detector.detectMultiScale(img, 1.3, 5)

    # Go through all faces.
    for (x, y, w, h) in faces:

        # Adjust accordingly if your webcam resolution is higher.
        if w > 100:
            # Draw rectangle to face.
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop detected face.
            detected_face = img[y:y+h, x:x+w]

            # Resize to 244x244.
            detected_face = cv2.resize(detected_face, (224, 224))

            # Convert image to numpy array.
            img_pixels = image.img_to_array(detected_face)

            # Expand its dimensionality for keras.
            img_pixels = np.expand_dims(img_pixels, axis=0)

            # Normalize your pixels between 0 and 1. Normalization will speed up training and avoid
            # gradient exposion.
            img_pixels /= 255

            # Pass the image to predictor in order to produce respresentation vector.
            captured_representation = model.predict(img_pixels)[0, :]

            found = 0

            # Go through picture database (all peoples faces dict) to compare each face.
            for i in all_people_faces:
                # This is the key of dictionary.
                person_name = 1

                # This is vector representaiton of faces of given name.
                representation = all_people_faces[i]

                # Compare the detected face to face in the database.
                similarity = findCosineSimilarity(representation, captured_representation)

                # If we find the match, attach persons name to face.
                if (similarity < 0.30):
                    cv2.putText(img, person_name[5:], (x + w + 15, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    found = 1

                    break

            if (found == 0):
                cv2.putText(img, 'Unknown', (x + w + 15, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cap.release()
cv2.destroyAllWindows()

