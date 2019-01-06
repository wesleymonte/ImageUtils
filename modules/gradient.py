import cv2
import numpy as np

def getGradientSobel(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelX = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1)
    sobelY = np.uint8(np.absolute(sobelY))
    combinedSobel = cv2.bitwise_or(sobelX, sobelY)
    return combinedSobel

def getGradientLaplacian(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(grayscale, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap