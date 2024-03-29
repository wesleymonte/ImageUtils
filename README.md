# ImagesUtils

ImageUtils is a set of functions for image processing, built on Python 3.6 using mainly OpenCV 3.4.

Example of functions: translation, resizing, thresholding, edge detection and among others.


# Installation

*  `git clone https://github.com/wesleymonte/ImageUtils.git`

*  `cd ImageUtils`

*  `pip install -r requirements.txt`


# Examples

Here are some examples of using functions

## Translation

#### Command:

<pre>
import imutils
import cv2

cv2.imshow("Original", image)
cv2.imshow("Shifted", imutils.translation(image, -60, -60))
cv2.waitKey(0)
</pre>

#### Output:

<img  src="images/translation.png?raw=true"  alt="Translation example"  style="max-width: 500px;">

## Gaussian Blur

#### Command:

<pre>
import imutils
import cv2

cv2.imshow("Original", image)
cv2.imshow("Gaussian Blur", imutils.GaussianBlur(image, 11))
cv2.waitKey(0)
</pre>

#### Output:

<img  src="images/gaussian_blur.png?raw=true"  alt="Gaussian Blur example"  style="max-width: 500px;">

## Adaptive Thresholding

#### Command:

<pre>
import imutils
import cv2

cv2.imshow("Original", image)
cv2.imshow("Adaptive Thresholding", imutils.adaptiveThresholding(image, 9, 3))
cv2.waitKey(0)
</pre>

#### Output:

<img  src="images/adaptive_thresholding.png?raw=true"  alt="Gaussian Blur example"  style="max-width: 500px;">