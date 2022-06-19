# VirtualFace.py

# CSCI-509 - AI & Computer Vision | Summer 2022 | USC Upstate
# Tue. 06/14/2022
# Trevor Reynen

# Align other peoples face onto your face through images.


# Imports.
import cv2
import dlib
import numpy as np
import os


# The pre-trained model that predicts the rectangles that correspond to facial features of a face.
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

# Points from the second image to overlay on the first. The convex hull of each element will
# be overlaid.
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]

# Amount of blur to use during color correction, as a fraction of the pupillary distance.
COLOR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


def get_landmarks(img):
    # Returns facial landmarks as (x, y) coordinates.
    rects = detector(img, 1)

    if len(rects) > 1:
        raise TooManyFaces

    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


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

    # Solve the procrustes problem by subtracting centroids, scaling by the standard deviation, and
    # then using the SVD to calculate the rotation. See the following for more details:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

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

    # The R we seek is in fact the transpose of the one given by U * Vt. This is because the above
    # formulation assumes the matrix goes on the right (with row vectors) where as our solution
    # requires the matrix to be on the left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def read_img_and_landmarks(image):
    img = image
    img = cv2.resize(img, None, 1, 1, cv2.INTER_LINEAR)
    img = cv2.resize(img, (img.shape[1] * SCALE_FACTOR, img.shape[0] * SCALE_FACTOR))
    s = get_landmarks(img)

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


# ==========<  Code above was given (I formatted), code below was from lab video.  >==========

# What is difficulty of virtual face or face swapping?
# Getting key landmarks of face (such as eye, nose, and mouth) aligned correctly.

# Major steps for face swapping:
#   1. Identify facial features (facial landmarks)
#   2. Wrap the second image to fit the new and different facial expression to the first image.
#      Rotating, scaling, and translating the second image to fit over the first.
#   3. Color matching of two faces.
#      Adjusting the color balance in the second image to match that of the first.
#   4. Create seamless borders on the edges of the new swapped face.
#      Blending features from the second image on top of the first.

# Face of image2 is placed onto image1.
# image1 is the base image.
def virtualFace(image1, image2):
    # 1. Using dlib to extract facial landmarks.
    img1, landmarks1 = read_img_and_landmarks(image1)
    img2, landmarks2 = read_img_and_landmarks(image2)

    # Produce the Transformation Matrix (M) that maps points from one face to the next.
    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

    # Produce image mask that outline which parts of image 2 will be overlaid on image 1.
    mask = get_face_mask(img2, landmarks2)

    # Produce warped mask (image 1 with image 2's face overlaid over it.)
    warped_mask = warp_img(mask, M, img1.shape)

    # Produce a combined mask which ensures the features from image 1 are covered up and features
    # for image 2 are visible.
    combined_mask = np.max([get_face_mask(img1, landmarks1), warped_mask], 0)

    warped_img2 = warp_img(img2, M, img1.shape)

    # Produce color corrected warped image 2 by matching skin tone and lighting between 2 images.
    warped_corrected_img2 = correct_colors(img1, warped_img2, landmarks1)

    # Apply mask to produce final image.
    output_img = img1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask

    cv2.imwrite(uniqueFile('./images/saved/VirtualFace.jpg'), output_img)


def uniqueFile(file):
    fName, fExt = os.path.splitext(file)
    i = 1
    saveFile = fName + '-' + str(i) + fExt

    while os.path.exists(saveFile):
        saveFile = fName + '-' + str(i) + fExt
        i += 1

    return saveFile


# Enter path of your input image here.
image1 = cv2.imread('./images/Hillary.jpg')
image2 = cv2.imread('./images/Trump.jpg')
image3 = cv2.imread('./images/Trev.jpg')

virtualFace(image1, image2) # Hillary & Trump
virtualFace(image2, image1) # Trump & Hillary
virtualFace(image3, image2) # Trev & Trump
virtualFace(image2, image3) # Trump & Trev

