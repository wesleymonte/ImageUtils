import cv2
import numpy as np
from modules import blur
from modules import mythresholding
from modules import gradient
from modules import edge_detection

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
    if width is None and height is None:
        return image
    if width is not None and height is None:
        r = width / image.shape[1]
        height = int(image.shape[0] * r)
    elif height is not None and width is None:
        r = height / image.shape[0]
        width = int(image.shape[1] * r)
        
    return cv2.resize(image, (width, height), inter)

def flip(image, flipCode=1):
    return cv2.flip(image, flipCode)

def crop(image, width1, width2, height1, height2):
    return image[height1:height2, width1:width2]

def AverageBlur(image, K):
    return blur.AverageBlur(image, K)

def GaussianBlur(image, K):
    return blur.GaussianBlur(image, K)

def MedianBlur(image, K):
    return blur.MedianBlur(image, K)

def BilateralBlur(image, K, C, S):
    return blur.BilateralBlur(image, K, C, S)

def thresholding(image, inv=False):
    return mythresholding.thresholding(image, inv)

def adaptiveThresholding(image, K, C, inv=False, method="gaussian"):
    """
    Apply the method Adaptive Thresholding in Image

    Parameters
    ----------
    image : array
        The image 
    K : integer
        Neighborhood Size
    C : integer
        Integer that is subtracted from the mean
    inv: boolean
        Invert thresholded image colors
    method: str
        Method that will be used in the thresholding
    
    Returns
    -------
    ndarray
        Thresholded Image
    """

    return mythresholding.adaptiveThresholding(image, K, C, inv, method)

def getGradientSobel(image):
    return gradient.getGradientSobel(image)

def getGradientLaplacian(image):
    return gradient.getGradientLaplacian(image)

def canny(image, K=5, lower=30, upper=150):
    return edge_detection.canny(image, K, lower, upper)

def auto_canny(image, sigma=0.33):
    return edge_detection.auto_canny(image, sigma)