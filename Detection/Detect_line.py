from __future__ import division
import cv2
import numpy as np

# get img and change it to grayscale
original_img = cv2.imread('test.png')
grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# change img to black & white
(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

(x, y) = im_bw.shape

# sums up the column pixel value of a given row in an image
def sumup_column(row):
	sum = 0
	for i in row:
		sum = sum + i
	return sum / row.size

# finds the lines that contain text and
# return them as a list of tuples that indicate the location of starting and ending pt of each line
def find_line(img):
	start_idx, end_idx = -1, -1
	trunc_row = []
	(row, _) = img.shape
	for i in range (0, row):
		sum = sumup_column(img[i])
		if sum > 0 and start_idx == -1:
			start_idx = i
		elif (sum == 0 and start_idx == -1) or sum > 0:
			continue
		elif sum == 0 and start_idx >= 0:
			end_idx = i
			if end_idx - start_idx >= 5:
				if start_idx >= 2 and end_idx <= row - 3:
					trunc_row.append((start_idx - 2, end_idx + 2))
				elif start_idx >= 2:
					trunc_row.append((start_idx - 2, end_idx))
				elif end_idx <= row - 3:
					trunc_row.append((start_idx, end_idx + 2))
				else:
					trunc_row.append((start_idx, end_idx))

			start_idx, end_dix = -1, -1

	return trunc_row

# truncates the imgs to line imgs
# returns list of img objects
def get_line_imgs(img, trunc_row):
	line_imgs = []
	l = len(trunc_row)
	for i in range (0, l):
		line_imgs.append(img[trunc_row[i][0]:trunc_row[i][1]]) 
	return line_imgs

cnt = 0
for i in get_line_imgs(im_bw, find_line(im_bw)):
	name = "line" + str(cnt) + ".png"
	cnt = cnt + 1
	cv2.imwrite(name, i)

