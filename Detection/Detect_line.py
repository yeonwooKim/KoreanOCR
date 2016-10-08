from __future__ import division
import cv2
import numpy as np

# get img and change it to grayscale
original_img = cv2.imread('test/test.png')
grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# change img to black & white
(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

(x, y) = im_bw.shape

# sums up the column pixel value of a given row in an image
def sumup_column(img, row_number):
	sum = 0
	(_, length) = img.shape
	for i in range(0, length):
		sum = sum + img.item(row_number, i)
	return sum / length

# sums up the row pixel value of a given column in an image
def sumup_row(img, col_number):
	sum = 0
	(length, _) = img.shape
	for i in range(0, length):
		sum = sum + img.item(i, col_number)
	return sum / length

# finds the lines that contain text and
# return them as a list of tuples that indicate the location of starting and ending pt of each line
def find_line(img):
	start_idx, end_idx = -1, -1
	trunc_row = []
	(row, _) = img.shape
	for i in range (0, row):
		sum = sumup_column(img, i)
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

def label_location(trunc_row):
	label, num_p, num_l = [], 0, 0
	length = len(trunc_row)
	for i in range (0, length - 1):
		label.append((num_p, num_l))
		num_l = num_l + 1
		if trunc_row[i + 1][0] - trunc_row[i][1] > 20:
			num_p = num_p + 1
			num_l = 0
	label.append((num_p, num_l))
	return label

# trims line images
def trim_line(img):
	(x, y) = img.shape
	fst, lst = -1, -1
	for i in range (0, y):
		sum = sumup_row(img, i)
		if sum > 0:
			fst = i
			break
	for i in range (0, y):
		sum = sumup_row(img, y - 1 - i)
		if sum > 0:
			lst = y - 1 - i
			break
	if fst < 10 and lst >= y - 10:
		return img
	elif fst < 10:
		return img[0:x, 0:lst + 10]
	elif lst >= y - 10:
		return img[0:x, fst - 10:y]
	return img[0:x, fst - 10:lst + 10]

# truncates the imgs to line imgs
# returns list of img objects
def get_line_imgs(img, trunc_row):
	line_imgs = []
	l = len(trunc_row)
	for i in range (0, l):
		line_imgs.append(trim_line(img[trunc_row[i][0]:trunc_row[i][1]]))
	return line_imgs

def save_line_imgs(img, trunc_row):
	length = len (trunc_row)
	label = label_location(trunc_row)
	imgs = get_line_imgs(img, trunc_row)
	for i in range (0, length):
		name = "test/paragraph" + str(label[i][0]) + "line" + str(label[i][1]) + ".png"
		cv2.imwrite(name, imgs[i])

save_line_imgs(im_bw, find_line(im_bw))
