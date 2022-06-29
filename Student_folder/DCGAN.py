# DCGAN.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/28/2022
# Trevor Reynen

# MNIST DCGAN - We're going to create a GAN that generates synthetic handwritten digits.


# Imports.
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Keras Imports.
from keras import backend as K, initializers
from keras.datasets import mnist
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam


K.set_image_dim_ordering('th')

# The dimensionality has been set at 100 for consistency with other GAN implementations, but
# 10 works better here.
latent_dim = 100

# Load MNIST data.
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
print(X_train.shape)

X_train = X_train[:, np.newaxis, :, :]
print(X_train.shape)

# Use Adam as the Optimizer.
adam = Adam(lr=0.0002, beta_1=0.5)

# Make our generator model.
generator = Sequential()

# Transforms the input into a 7 × 7 128-channel feature map.
generator.add(Dense(128 * 7 * 7, input_dim=latent_dim))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))

# Produces a 28 × 28 1-channel feature map (shape of a MNIST image).
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
print(generator.summary())
generator.compile(loss='binary_crossentropy', optimizer=adam)

# Make discriminator model.
discriminator = Sequential()

discriminator.add(Conv2D(64,
                         kernel_size=(5, 5),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=initializers.RandomNormal(stddev=0.02),
                         input_shape=(1, 28, 28)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
print(discriminator.summary())
discriminator.compile(optimizer=adam, loss='binary_crossentropy')


# Creating the adversarial N\network. We need to make the discriminator weights non trainable. This
# only applies to the GAN model.
discriminator.trainable = False
ganInput = Input(shape=(latent_dim,))
x = generator(ganInput) # x is generated image.
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(optimizer=adam, loss='binary_crossentropy')

# The discriminator and generator losses.
dLosses = []
gLosses = []


# Plot the loss from each batch.
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))

    plt.plot(dLosses, label='Discriminative loss')
    plt.plot(gLosses, label='Generative loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    plt.savefig('./assets/GAN/ganimages/dcgan_loss_epoch_%d.png' % epoch)


# Create a wall of generated MNIST images.
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generatedImages = generator.predict(noise)

    plt.figure(figsize=figsize)

    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()

    plt.savefig('./assets/GAN/ganimages/dcgan_generated_image_epoch_%d.png' % epoch)


# Save the generator and discriminator networks (and weights) for later use.
def saveModels(epoch):
    generator.save('./assets/GAN/ganmodels/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('./assets/GAN/ganmodels/dcgan_discriminator_epoch_%d.h5' % epoch)


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# Introduction to Generative Adversarial Networks (GAN)
# GAN can generate realistic artificial images that could be near indistinguishable from real ones.
# Architectures of GANs.
# GANs consist of generator and discriminator.
# Generator takes a random noisy image and decode it into a synthetic image.
# Discriminator is used to predict whether images are real or synthetic (generated).
# The generator tries to make better fakes to beat the Discriminator while the Discriminator learns
# to get better at spotting fakes.
# In the work, GAN is used to create fake handwritten digits.

# Train our GAN and Plot the Synthetic Image Outputs.
# After each consecutive Epoch we can see how synthetic image being improved.
epochs = 10
batchSize = 128
batchCount = X_train.shape[0] / batchSize

print('Epochs:', epochs)
print('Batch size:', batchSize)
print('Batches per epoch:', batchCount)

for e in range(1, epochs + 1):
    print('-' * 15, 'Epoch %d' % e, '-' * 15)

    for i in tqdm(range(int(batchCount))):
        # Get a random set of input noise.
        noise = np.random.normal(0, 1, size=[batchSize, latent_dim])

        # Get a batch of real images.
        imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

        # Generate fake MNIST images.
        generatedImages = generator.predict(noise)

        # Combine fake images and real images.
        X = np.concatenate([imageBatch, generatedImages])

        # Produce Labels for generated and real data.
        yDis = np.zeros(2 * batchSize)

        # Create one-sided label smoothing to prevent attacking.
        yDis[:batchSize] = 0.9

        # Train discriminator. train_on_batch runs a single gradient update on one batch of data.
        # X is data, yDis is label, dloss is loss for discriminator.
        discriminator.trainable = True
        dloss = discriminator.train_on_batch(X, yDis)

        # Train generator. Create random noise.
        noise = np.random.normal(0, 1, size=[batchSize, latent_dim])

        # Create label for generator.
        yGen = np.ones(batchSize)

        # Freeze weights for discriminator.
        discriminator.trainable = False

        # Train generator. gloss is loss for generator.
        gloss = gan.train_on_batch(noise, yGen)

    # Store loss of most recent batch from this epoch.
    dLosses.append(dloss)
    gLosses.append(gloss)

    # Plotting and save model for every iteration.
    plotGeneratedImages(e)

    # Plot losses from every epoch.
    plotLoss(e)
    saveModels(e)

