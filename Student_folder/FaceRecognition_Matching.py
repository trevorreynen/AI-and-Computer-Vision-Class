# FaceRecognition_Matching.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Wed. 06/29/2022
# Trevor Reynen

# NOTE: The model needed for the labs FaceRecognition.py and FaceRecognition_Matching.py, called
# "vgg_face_weights.h5" is over 100 MB. For now, I will not be uploading it to GitHub inside this
# repository. If I find a link to the model I will either add it here or add code to the two
# FaceRecognition programs to download the model if it isn't already.

# This code is used to compare the similarity of 2 faces.
# The VGG-Face CNN descriptors are computed using our CNN implementation based on the
# VGG-Very-Deep-16 CNN architecture.


# Imports.
import matplotlib.pyplot as plt
import numpy as np

# Keras Imports.
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Convolution2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img


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

# This is the threshold to decide whether two faces are from the same person.
epsilon = 0.40

# Image in the lab is stored in training_faces folder.

# verifyFace function will compare vector representation of two faces, calculate their distance,
# and decide whether two faces are from the same person or not.
def verifyFace(img1, img2):
    img1_process = preprocess_image('./assets/images/face_recognition/training_faces/%s' % img1)

    # We obtain vector representaiton for img1. We get row0 of the prediction result.
    img1_representation = model.predict(img1_process)[0, :]

    # We obtain vector representation for img2.
    img2_process = preprocess_image('./assets/images/face_recognition/training_faces/%s' % img2)

    # We obtain vector representaiton for img1. We get row0 of the prediction result.
    img2_representation = model.predict(img2_process)[0, :]

    # Calculate vector distance between two faces.
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)

    print('Cosine Similarity:', cosine_similarity)

    # If the vector distance is less than given threshold, they are the same person.
    if (cosine_similarity < epsilon):
        print('They are the same person')
    else:
        print('They are not the same person.')

    print()

    # Show two faces side by side.
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image.load_img('./assets/images/face_recognition/training_faces/%s' % img1))
    plt.xticks([]); plt.yticks(([]))

    f.add_subplot(1, 2, 2)
    plt.imshow(image.load_img('./assets/images/face_recognition/training_faces/%s' % img2))
    plt.xticks([]); plt.yticks(([]))
    plt.show(block=True)


# Let us call verifyFace function 4 times to compare 4 pairs of faces.
verifyFace('angelina.jpg', 'angelina2.jpg')
verifyFace('angelina.jpg', 'Monica.jpg')
verifyFace('angelina.jpg', 'Rachel.jpg')
verifyFace('angelina.jpg', 'angelina3.jpg')

