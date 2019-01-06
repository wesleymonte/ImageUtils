import cv2
import mahotas

def thresholding(image, inv=False):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_type = cv2.THRESH_BINARY
    if inv:
        thresh_type = cv2.THRESH_BINARY_INV
    
    T = mahotas.otsu(grayscale)
    (_, thresh) = cv2.threshold(grayscale, T, 255, thresh_type)
    return thresh

def adaptiveThresholding(image, K, C, inv=False, method="gaussian"):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_type = cv2.THRESH_BINARY
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    if inv:
        thresh_type = cv2.THRESH_BINARY_INV
    if method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    thresh = cv2.adaptiveThreshold(grayscale, 255, adaptive_method, thresh_type, K, C)
    return thresh