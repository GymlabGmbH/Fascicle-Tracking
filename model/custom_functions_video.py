from __future__ import division
import os
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")



from skimage.transform import resize
from skimage.morphology import skeletonize
from scipy.signal import resample, savgol_filter, butter, filtfilt
from PIL import Image, ImageDraw
import cv2

import tensorflow as tf

from keras import backend as K
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Intersection over union (IoU), a measure of labelling accuracy (sometimes also called Jaccard score)
def IoU(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


# Function to sort contours from proximal to distal (the bounding boxes are not used)
def sort_contours(cnts):
    # initialize the reverse flag and sort index
    i = 1
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=False))

    return (cnts, boundingBoxes)


# Find only the coordinates representing one edge of a contour. edge: T (top) or B (bottom)
def contour_edge(edge, contour):
    pts = list(contour)
    ptsT = sorted(pts, key=lambda k: [k[0][0], k[0][1]])
    allx = []
    ally = []
    for a in range(0, len(ptsT)):
        allx.append(ptsT[a][0, 0])
        ally.append(ptsT[a][0, 1])
    un = np.unique(allx)
    # sumA = 0
    leng = len(un) - 1
    x = []
    y = []
    for each in range(5, leng - 5):  # Ignore 1st and last 5 points to avoid any curves
        indices = [i for i, x in enumerate(allx) if x == un[each]]
        if edge == 'T':
            loc = indices[0]
        else:
            loc = indices[-1]
        x.append(ptsT[loc][0, 0])
        y.append(ptsT[loc][0, 1])
    return np.array(x), np.array(y)


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False




# Function to compute the distance between 2 x,y points
def distFunc(x1, y1, x2, y2):
    xdist = (x2 - x1) ** 2
    ydist = (y2 - y1) ** 2
    return np.sqrt(xdist + ydist)
