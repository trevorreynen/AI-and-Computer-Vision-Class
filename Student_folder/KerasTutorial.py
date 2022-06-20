# KerasTutorial.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/21/2022
# Trevor Reynen

# This is a tutorial for Keras.


# Imports.
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Keras Imports.
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils


# Homework:
# Finish the training process and get mnist_simple_cnn.h5 model.
# Save the final plot for Validation Accuracy and Training Accuracy.


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# Load the MNIST dataset.
# x_train is the feature in the training data.
# y_train is the label in the training dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Shape of x_train (60000, 28, 28).
# Shape of y_train(60000,).
# Our image dimensions are 28 x 28, with no color channels.
print('Shape of x_train', x_train.shape)
print('Shape of y_train', y_train.shape)

# Let us take a look at some of images in this dataset. Use OpenCV to display 6 random images
# from dataset.
for i in range(0, 6):
	random_num = np.random.randint(0, len(x_train))
	img = x_train[random_num]
	window_name = 'Random Sample #' + str(i)
	cv2.imshow(window_name, img)
	cv2.waitKey(1000)

cv2.destroyAllWindows()


# Lets store the number of rows and columns.
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
print('img_rows: ', img_rows)
print('img_cols: ', img_cols)

# Prepare our dataset for training.
# Keras require the data format: Number of Samples, Rows, Cols, Depth.
# Getting our data in the right 'shape' needed for Keras.
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Store the shape of a single image.
input_shape = (img_rows, img_cols, 1)

# Change our image type to float32 datatype
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1).
x_train /= 255
x_test /= 255

# One Hot Encode Our Labels (Y)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# Create our CNN Model.
# Create Sequential Model, which is a linear stack of layers.
model = Sequential()

# Creating our convolution layer.
# The first conv layer uses 32 filters of size 3 x 3.
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# The second conv layer uses 64 filters of size 3 x 3.
model.add(Conv2D(64, (3, 3), activation='relu'))

# Then we downscale our data by 2 x 2 and apply a dropout with rate of 0.25.
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten our Max Pool output layer.
model.add(Flatten())

# Create 128 unit Dense layer.
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Create FC/Dense output layer with the 10 categorical units. Softmax function will provide probability for each class.
model.add(Dense(num_classes, activation='softmax'))

# Compiling simply creates an object that stores our model we have created.
# We can specify our loss algorithm, optimizer and define our performance metrics.
model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])

print(model.summary())


# Train the model.
# Training Parameters - Batch size is how many image we will process per batch.
batch_size = 128
epochs = 10

# We place our formatted data as the inputs and set the batch size, number of epochs.
# We store our model's training results for plotting in future.
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# We then use Keras' model.evaluate function to output the model's final performance.
score = model.evaluate(x_test, y_test, verbose=0)

# Here we are examining Test Loss and Test Accuracy.
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# Save the model.
model.save('./models/mnist_simple_cnn.h5')
print('Model Saved')


# Plotting the loss and accuracy charts.
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label='Validation Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label='Validation Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

