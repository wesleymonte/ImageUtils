import cv2
import numpy as np

def AverageBlur(image, K):
    return cv2.blur(image, (K, K))

def GaussianBlur(image, K):
    return cv2.GaussianBlur(image, (K, K), 0)

def MedianBlur(image, K):
    return cv2.medianBlur(image, K)

def BilateralBlur(image, K, C, S):
    return cv2.bilateralFilter(image, K, C, S)