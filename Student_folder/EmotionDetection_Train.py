# EmotionDetection_Train.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Wed. 06/22/2022
# Trevor Reynen

# We will detect six emotions including Angry, Fear, Happy, Neutral, Sad, Surprised.

# NOTE: We were told to delete the "Disgust" folder from both ./fer2013/train/
# and ./fer2013/validation/. According to my teacher, those "Disgust" folders have less samples to
# train the model with. So, it will unbalance the trained model and possibly cause problems.
# Those Disgust files were moved to ./unused-assets/fer2013/...

# Using LittleVGG for Emotion Detection.
# In the fer2013 dataset folder, we have train and validation folder. In both folders, we will
# delete disgust directory, which only has 400 samples.


# Keras Imports
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 16

# Keras LittleVGG Model.
model = Sequential()

model.add(Conv2D(32, (3, 3),
                 padding='same',
                 kernel_initializer='he_normal',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3),
                 padding='same',
                 kernel_initializer='he_normal',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block #2: second CONV => RELU => CONV => RELU => POOL. Layer set.
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block #3: third CONV => RELU => CONV => RELU => POOL. Layer set.
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block #4: third CONV => RELU => CONV => RELU => POOL. Layer set.
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Block #5: first set of FC => RELU layers.
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block #6: second set of FC => RELU layers.
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block #7: softmax classifier.
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

train_data_dir = './fer2013/train'
validation_data_dir = './fer2013/validation'


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# Try some data augmentation to increase training datasize.
train_datagen = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.4,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   fill_mode='nearest',
                                   horizontal_flip=True,
                                   rescale=1./255)

# For validation dataset, we only use rescaling function.
validation_datagen = ImageDataGenerator(rescale=1./255)

# Takes the path to a directory & generates batches of augmented data.
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_rows, img_cols),
                                                    color_mode='grayscale',
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_rows, img_cols), color_mode='grayscale',
                                                              class_mode='categorical',
                                                              batch_size=batch_size,
                                                              shuffle=True)

# Save the model after every epoch.
checkpoint = ModelCheckpoint('./models/trained/emotion_little_vgg_3.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

# Stop training when a monitored quantity has stopped improving.
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop, checkpoint, reduce_lr]

# Configures the model for training.
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

nb_train_samples = 28273
nb_validation_samples = 3534
# After 30 epochs, it will reach 60%.
# After 10 epochs, it will reach 50%.
epochs = 35

# Trains the model on data generated batch-by-batch by a Python generator.
history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples//batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              validation_data=validation_generator,
                              validation_steps=nb_validation_samples//batch_size)


# Homework:
# Produce /models/trained/emotion_little_vgg_3.h5 model.

