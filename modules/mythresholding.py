import cv2
import mahotas
from modules.blur import GaussianBlur

def thresholding(image, inv=False):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY
    if inv:
        thresh_type = cv2.THRESH_BINARY_INV
    blurred = GaussianBlur(grayscale, 7)
    T = mahotas.otsu(blurred)
    thresh = cv2.threshold(blurred, T, 255, thresh_type)[1]
    return thresh

def adaptiveThresholding(image, K, C, inv=False, method="gaussian"):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    if inv:
        thresh_type = cv2.THRESH_BINARY_INV
    if method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    blurred = GaussianBlur(grayscale, 7)
    thresh = cv2.adaptiveThreshold(blurred, 255, adaptive_method, thresh_type, K, C)
    return thresh