# NeuralStyleTransfer.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/28/2022
# Trevor Reynen

# Neural Style Transfer in Keras.

# Let's first define our Target and Style Reference Images.
# Target is our photo we wish to apply some artistic style too.
# Style reference is the image of artist style we want to copy.


# Imports.
import imageio, time
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# Keras Imports.
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import img_to_array, load_img


ntNmae = 'es'

# Directory for target image and style image you need create style_transfer_results to hold final
# generated images.
# The target image to transform.
target_image_path = './assets/images/Trev.jpg'

# The style image.
style_reference_image_path = './assets/images/starrynight.jpg'

result_prefix = style_reference_image_path.split('./assets/images/')[1][:-4] + '_onto_' + target_image_path.split('./assets/images/')[1][:-4]

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)


def preprocess_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)

    return img


def deprocess_image(x):
    # Remove zero-center by mean pixel.
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR' -> 'RGB'.
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x

# Our Loss Functions and Gram Matrix.
# Content Loss. Gram Matrix. Style Loss. Total Variation Loss.

def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# gram_matrix focuses on how to get the feature out of the style image.
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))

    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width

    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1,      :])
    b = K.square(x[:, :img_height-1, :img_width-1, :] - x[:,     :img_height-1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width

    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1,      :])
    b = K.square(x[:, :img_height-1, :img_width-1, :] - x[:,     :img_height-1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))


# Loading our VGG16 Model and then applying it to our Target, Style and Generated Image.
target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# This placeholder will contain our generated image.
combination_image = K.placeholder((1, img_height, img_width, 3))

# We combine the 3 images into a single batch
input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

# We build the VGG16 network with our batch of 3 images as input. The model will be loaded with pre-trained ImageNet weights.
model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

print(model.summary())
print('Model Loaded')

# This is the final loss we will be minimizing.
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Name of layer used for content loss.
content_layer = 'block5_conv2'

# Name of layers used for style loss.
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Weights in the weighted average of the loss components.
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# Define the loss by adding all components to a `loss` variable.
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

# Create our Gradient Descent Process.
# Get the gradients of the generated image wrt the loss.
grads = K.gradients(loss, combination_image)[0]

# Function to fetch the values of the current loss and the current gradients.
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_values = None


    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values

        return self.loss_value


    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None

        return grad_values


evaluator = Evaluator()


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# What is Neural Style Transfer?
# It enables the artistic style of an image (Style images) to be applied to another image (target
# image). In other words, we want to preserve the content of the original (target image) while
# adopting the style of the reference image. Artistic style means the color patterns, combination
# and brush strokes of style images.

# What is the process?
# We keep changing pixel values as to optimize a cost function (keeping our pre-trained CNN weights
# constant). The cost function consists of a content loss and a style loss. Content loss can make
# sure our generated image retains a similar look to the original image. Style loss ensure our
# textures between images look similar. In the work, we use pre-trained CNN (VGG) for Neural Style
# Transfer.

# After 10 iterations little change occurs.
iterations = 10

# This is our initial state: the target image.
x = preprocess_image(target_image_path)
# Note that 'scipy.optimize.fmin_l_bfgs_b' can only process flat vectors.
x = x.flatten()

# Run our style transfer loop.
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()

    # Run scipy-based optimized (L-BFGS) over the pixels of the generated image.
    # So as to minimize the neural style loss.
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)

    print('Current loss value:', min_val)

    # Save current generated target image.
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = './assets/images/saved/neural_style_transfer/' + result_prefix + '_at_iteration_%d.png' % i
    imageio.imwrite(fname, img)

    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


# Display content image.
plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()

# Display style image.
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()

# Dispaly generated image.
plt.imshow(img)
plt.show()

