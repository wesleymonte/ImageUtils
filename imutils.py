import cv2
import numpy as np
import argparse

def getShape(image):
    return (image.shape[1], image.shape[0])

def getCenter(image):
    return (image.shape[1] // 2, image.shape[0] // 2)

def translation(image, X, Y):
    m = np.float32([[1, 0, X], [0, 1, Y]])
    shifted = cv2.warpAffine(image, m, getShape(image))
    return shifted

def rotation(image, degrees, scale=1.0):
    m = cv2.getRotationMatrix2D(getCenter(image), degrees, scale)
    rotated = cv2.warpAffine(image, m, getShape(image))
    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    if width is not None and height is not None:
        return image
    if width is not None:
        r = width / image.shape[1]
        height = int(image.shape[0] * r)
    else:
        r = height / image.shape[0]
        width = int(image.shape[1] * r)
        
    return cv2.resize(image, (width, height), inter)

def flip(image, flipCode=1):
    return cv2.flip(image, flipCode)

def crop(image, width1, width2, height1, height2):
    return image[height1:height2, width1:width2]