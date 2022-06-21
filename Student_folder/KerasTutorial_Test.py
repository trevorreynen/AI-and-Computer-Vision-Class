# KerasTutorial_Test.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/21/2022
# Trevor Reynen

# This will test our model performance.

# Imports.
import cv2
import numpy as np

# Keras Imports.
from keras.datasets import mnist
from keras.models import load_model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

classifier = load_model('./models/mnist_simple_cnn.h5')

# pred is the prediction results of image. input_img is the input image.
def draw_test(name, pred, input_img):
    BLACK = [0, 0, 0]

    # Build border to the right of image with the size imageL.shape[0].
    expanded_img = cv2.copyMakeBorder(input_img, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)

    expanded_img = cv2.cvtColor(expanded_img, cv2.COLOR_GRAY2BGR)

    cv2.putText(expanded_img, str(pred), (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    cv2.imshow(name, expanded_img)


# Let us input some of our validation data into our classifier.
for i in range(0, 20):
    rand = np.random.randint(0, len(x_test))
    input_img = x_test[rand]
    #cv2.imshow('Input Image', input_img)
    # Shape of input_img (28, 28).
    #print('Shape of Input Image (input_img):', input_img.shape)

    # Option 2 (Factor with float value): Enlarge image for displaying.
    imageL = cv2.resize(input_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('Enlarged Image', imageL)
    # Shape of imageL (112, 112).
    #print('Shape of Large Image (imageL):', imageL.shape)
    input_img = input_img.reshape(1, 28, 28, 1)

    # Get Prediction.
    res = str(classifier.predict_classes(input_img, 1, verbose=0)[0])

    draw_test('Prediction', res, imageL)
    cv2.waitKey(1000)

cv2.destroyAllWindows()

