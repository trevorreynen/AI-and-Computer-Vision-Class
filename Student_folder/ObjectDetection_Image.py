# ObjectDetection_Image.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Thu. 06/28/2022
# Trevor Reynen


# Object_Detection_image_v3_template.py
# We will use a pre-trained model to detect objects in an image.


# Imports.
import cv2
from distutils.version import StrictVersion
import numpy as np
from object_detection.utils import label_map_util, ops as utils_ops, visualization_utils as vis_util
import os
from PIL import Image
import six
import tarfile
import tensorflow as tf


if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = './assets/models/pretrained/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './assets/labels/mscoco_label_map.pbtxt'

# Download Model.
opener = six.moves.urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, './assets/models/pretrained/' + MODEL_FILE)
tar_file = tarfile.open('./assets/models/pretrained/' + MODEL_FILE)

for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)

    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, './assets/models/pretrained/')

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map.
# Label maps map indices to category names, so that when our convolution network predicts 5, we
# know that this corresponds to airplane. Here we use internal utility functions, but anything that
# returns a dictionary mapping integers to appropriate string labels would be fine.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print(category_index)
print()


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# Detection.
# For the sake of simplicity we will use only 2 images: image1.jpg. image2.jpg.
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = './assets/images/test_images/'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 13)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors.
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = { output.name for op in ops for output in op.outputs }
            tensor_dict = {}

            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image.
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks,
                                                                                      detection_boxes,
                                                                                      image.shape[1],
                                                                                      image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                # Follow the convention by adding back the batch dimension.
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference.
            output_dict = sess.run(tensor_dict, feed_dict={ image_tensor: image })

            # All outputs are float32 numpy arrays, so convert types as appropriate.
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========
# NOTE: I changed a few things in the code above (given code by teacher). I needed to change
# anything related to pathing as I stuctured my project myself and not the way my teacher intended.
# I have also worked on TFOD quite a bit before this lab, so I know what most of this does. I had a
# project in my senior seminar class that was on TFOD.


# Go through each image in test_images folder.
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)

    # Convert image into numpy array.
    image_np = load_image_into_numpy_array(image)

    # Converts image from BGR to RGB.
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    #cv2.imshow('Image', image_np)
    #cv2.waitKey(2000)

    # Expand dimensions of numpy array.
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Run object detection algorithm.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

    print('Output Dict:')
    print(output_dict)
    print()

    # Visualization of results of a detection.
    # Each box represents a part of the image where a particular object was detected. Each score
    # represents the level of confidence for each of the objects. Score is shown on the result
    # image, together with the class label.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4)

    cv2.imshow('Image', image_np)
    cv2.waitKey(3000)

