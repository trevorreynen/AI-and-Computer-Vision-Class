# VirtualFaceLive.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/14/2022
# Trevor Reynen

# Align other peoples face onto your webcam image.

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
COLOUR_CORRECT_BLUR_FRAC = 0.6
cascade_path = './Haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im, dlibOn):
    if dlibOn == True:
        rects = detector(im, 1)

        if len(rects) > 1:
            return 'error'

        if len(rects) == 0:
            return 'error'

        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    else:
        rects = cascade.detectMultiScale(im, 1.3, 5)

        if len(rects) > 1:
            return 'error'

        if len(rects) == 0:
            return 'error'

        x, y, w, h = rects[0]
        rect = dlib.rectangle(x, y, x + w, y + h)

        return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()

    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (0, 0, 255))
        cv2.circle(im, pos, 3, (0, 255, 255))

    return im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], 1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


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


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, None, 0.35, 0.35, cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im, dlibOn)

    return im, s


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), output_im, cv2.BORDER_TRANSPARENT, cv2.WARP_INVERSE_MAP)

    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS], 0) - np.mean(landmarks1[RIGHT_EYE_POINTS], 0))
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)


def virtual_face(img, name):
    s = get_landmarks(img, True)

    if s == 'error':
        print('No or too many faces')

        return img

    im1, landmarks1 = img, s
    im2, landmarks2 = read_im_and_landmarks(name)

    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], 0)

    warped_im2 = warp_im(im2, M, im1.shape)

    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    # output_im is no longer in the expected OpenCV format so we use openCV to write the image to hard disk and then reload it.
    cv2.imwrite('output.jpg', output_im)
    image = cv2.imread('output.jpg')

    frame = cv2.resize(image, None, 1.5, 1.5, cv2.INTER_LINEAR)

    return image


# ===== Start Code Here =====

# dlibOn controls if use dlib's facial landmark detector (better) or use HAAR Cascade Classifiers (faster)
# Our base image is taken from webcam. Alight filter_image onto the webcam image.

#filter_image = './images/single-face1.jpg'
filter_image = './images/single-face2.jpg'

cap = cv2.VideoCapture(0)

dlibOn = False
while True:
    ret, frame = cap.read()

    # Reduce image size by 75% to reduce processing time and improve framerates.
    frame = cv2.resize(frame, None, 0.75, 0.75, cv2.INTER_LINEAR)

    # Flip image so that it's more mirror like.
    frame = cv2.flip(frame, 1)

    cv2.imshow('Our amazing virtual face', virtual_face(frame, filter_image))

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key.
        break

cap.release()
cv2.destroyAllWindows()

