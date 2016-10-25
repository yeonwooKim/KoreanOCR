import cv2
import numpy as np

img = cv2.imread('test/line_testing/test3.png')
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(x, y) = grayscale_img.shape
for i in range (0, x):
	for j in range (0, y):
		grayscale_img[i][j] = 255 - grayscale_img[i][j]

cv2.imwrite('test/test3.png', grayscale_img)
