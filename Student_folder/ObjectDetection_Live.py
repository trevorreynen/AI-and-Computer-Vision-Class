# ObjectDetection_Live.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Thu. 06/28/2022
# Trevor Reynen


# Object_Detection_live_template.py
# Live Object Detection from webcam.

# Imports.
import cv2
from distutils.version import StrictVersion
import numpy as np
from object_detection.utils import label_map_util, ops as utils_ops, visualization_utils as vis_util
import os
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


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========
# NOTE: I changed a few things in the code above (given code by teacher). I needed to change
# anything related to pathing as I stuctured my project myself and not the way my teacher intended.
# I have also worked on TFOD quite a bit before this lab, so I know what most of this does. I had a
# project in my senior seminar class that was on TFOD.


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./assets/images/dashcam2.mp4')

with detection_graph.as_default():
    # Configure our machine.
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = False

    # Set up the tensor for boxes, scores, and classes.
    with tf.Session(graph=detection_graph, config=config) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        #while True: # Use when cap is camera. If cap is video, use 'while cap.isOpened():'.
        while cap.isOpened(): # Use when cap is video. If cap is camera, use 'while True:'.
            # Read each frame.
            ret, image_np = cap.read()

            # Expand the dimentionality of numpy array.
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Run object detection.
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                     feed_dict={ image_tensor: image_np_expanded })

            # Visualize the boxes, scores, and classes for detection results.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)

            cv2.imshow('MobileNet SSD - Object Detection', image_np)

            if cv2.waitKey(1) == 13:  # 13 is the Enter key.
                break

cap.release()
cv2.destroyAllWindows()

