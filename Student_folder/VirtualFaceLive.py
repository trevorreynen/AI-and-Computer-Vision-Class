# VirtualFaceLive.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/14/2022
# Trevor Reynen

# Align other peoples face onto your face through webcam and image.

# Imports.
import cv2          # OpenCV is a library that has several hundreds of computer vision algorithms.
import dlib         # dlib is a toolkit for making machine learning and data analysis applications.
import numpy as np  # NumPy is an important library used for numerical computing.
import sys          # sys is used to manipulate different parts of the Python runtime environment.
import os           # os provides functions for interacting with the operating system.
from time import sleep


# Our pretrained model that predicts the rectangles that correspond to the facial features of a face.
PREDICTOR_PATH = './images/shape_predictor_68_face_landmarks.dat'
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each element will be overlaid.
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]

# Amount of blur to use during colour correction, as a fraction of the pupillary distance.
COLOR_CORRECT_BLUR_FRAC = 0.6
cascade_path = './Haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(img, dlibOn):
    # Returns facial landmarks as (x, y) coordinates.
    if dlibOn == True:
        rects = detector(img, 1)

        if len(rects) > 1:
            return 'error'

        if len(rects) == 0:
            return 'error'

        return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    else:
        rects = cascade.detectMultiScale(img, 1.3, 5)

        if len(rects) > 1:
            return 'error'

        if len(rects) == 0:
            return 'error'

        x, y, w, h = rects[0]
        rect = dlib.rectangle(x, y, x + w, y + h)

        return np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])


def annotate_landmarks(img, landmarks):
    # Overlays the landmark points on the image itself.
    img = img.copy()

    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (0, 0, 255))
        cv2.circle(img, pos, 3, (0, 255, 255))

    return img


def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color)


def get_face_mask(img, landmarks):
    img = np.zeros(img.shape[:2], np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(img, landmarks[group], 1)

    img = np.array([img, img, img]).transpose((1, 2, 0))
    img = (cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return img


def transformation_from_points(points1, points2):
    # Return an affine transformation [s * R | T] such that:
    #     sum ||s*R*p1,i + T - p2,i||^2
    # is minimized.

    # Solve the procrustes problem by subtracting centroids, scaling by the standard deviation, and then using the SVD to calculate the rotation. See the following for more details: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, 0)
    c2 = np.mean(points2, 0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This is because the above formulation assumes the matrix goes on the right (with row vectors) where as our solution requires the matrix to be on the left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def read_img_and_landmarks(fname):
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, 0.35, 0.35, cv2.INTER_LINEAR)
    img = cv2.resize(img, (img.shape[1] * SCALE_FACTOR, img.shape[0] * SCALE_FACTOR))
    s = get_landmarks(img, dlibOn)

    return img, s


def warp_img(img, M, dshape):
    output_img = np.zeros(dshape, dtype=img.dtype)
    cv2.warpAffine(img, M[:2], (dshape[1], dshape[0]), dst=output_img, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)

    return output_img


def correct_colors(img1, img2, landmarks1):
    blur_amount = COLOR_CORRECT_BLUR_FRAC * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS], 0) - np.mean(landmarks1[RIGHT_EYE_POINTS], 0))
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)


def virtual_face(img, name):
    s = get_landmarks(img, True)

    if s == 'error':
        print('No or too many faces')

        return img

    img1, landmarks1 = img, s
    img2, landmarks2 = read_img_and_landmarks(name)

    # Produce the Transformation Matrix (M) that maps points from one face to the next.
    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

    # Produce image mask that outline which parts of image 2 will be overlaid on image 1.
    mask = get_face_mask(img2, landmarks2)

    # Produce warped mask (image 1 with image 2's face overlaid over it.)
    warped_mask = warp_img(mask, M, img1.shape)

    # Produce a combined mask which ensures the features from image 1 are covered up and features for image 2 are visible.
    combined_mask = np.max([get_face_mask(img1, landmarks1), warped_mask], 0)

    warped_img2 = warp_img(img2, M, img1.shape)

    # Produce color corrected warped image 2 by matching skin tone and lighting between 2 images.
    warped_corrected_img2 = correct_colors(img1, warped_img2, landmarks1)

    # Apply mask to produce final image.
    output_img = img1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask

    # output_img is no longer in the expected OpenCV format so we use openCV to write the image to hard disk and then reload it.
    fileName = cv2.imwrite(uniqueFile('./images/saved/VirtualFaceLive.jpg'), output_img)
    image = cv2.imread(fileName)

    frame = cv2.resize(image, None, 1.5, 1.5, cv2.INTER_LINEAR)

    return image


def uniqueFile(file):
    fName, fExt = os.path.splitext(file)
    i = 1
    saveFile = fName + '-' + str(i) + fExt

    while os.path.exists(saveFile):
        saveFile = fName + '-' + str(i) + fExt
        i += 1

    return saveFile


# dlibOn controls if use dlib's facial landmark detector (better) or use HAAR Cascade Classifiers (faster)
# Our base image is taken from webcam. Alight filter_imgage onto the webcam image.

filter_image = './images/single-face1.jpg'
#filter_image = './images/single-face2.jpg'

dlibOn = False
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Reduce image size by 75% to reduce processing time and improve framerates.
    frame = cv2.resize(frame, None, 0.75, 0.75, cv2.INTER_LINEAR)

    # Flip image so that it's more mirror like.
    frame = cv2.flip(frame, 1)

    cv2.imshow('Virtual Face Live', virtual_face(frame, filter_image))

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cap.release()
cv2.destroyAllWindows()

