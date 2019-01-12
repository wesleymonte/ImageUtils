import cv2
from modules import blur
import numpy as np

def canny(image, K=5, lower=30, upper=150):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = blur.GaussianBlur(grayscale, K)
    return cv2.Canny(blurred, lower, upper)

def auto_canny(image, sigma=0.33):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(grayscale)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(grayscale, lower, upper)
    return edged