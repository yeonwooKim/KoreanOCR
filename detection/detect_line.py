from __future__ import division
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import statistics
from util import *

#if __name__ == '__main__':
	# Implemented for testing; get img and change it to grayscale
	#original_img = cv2.imread('test/line_testing/test2.png')
	#grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

	# Implemented for testing; change img to black & white
	#(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#(x, y) = im_bw.shape

# Finds the lines that contain text and
# Return them as a list of tuples that indicate the location of starting and ending pt of each line
def find_line(img):
	start_idx, end_idx = -1, -1
	trunc_row = []
	(row, _) = img.shape

	blank_thd = 2000 # Fix this to adjust blank line threshold
	for i in range (0, row):
		sum = sumup_row(img, i)
		if sum >= blank_thd and start_idx == -1:
			start_idx = i
		elif (sum < blank_thd and start_idx == -1) or sum > blank_thd:
			continue
		elif sum < blank_thd and start_idx >= 0:
			end_idx = i
			if end_idx - start_idx >= 10:
				trunc_row.append((start_idx, end_idx))
			start_idx, end_idx = -1, -1
	
	if start_idx != -1:
		trunc_row.append((start_idx, row - 1))

	return trunc_row

# Returns pair of total number of paragraphs and 
# list of labels containing paragraph #
def label_paragraph(trunc_row, height):
	label, num = [], 0
	length = len(trunc_row)

	para_thd = height # Fix this to adjust paragraph threshold
	for i in range (0, length - 1):
		label.append(num)
		if trunc_row[i + 1][0] - trunc_row[i][1] > para_thd:
			num = num + 1
	label.append(num)
	return (num + 1, label)

# Truncates the paragraph to lines
# Returns list of Line objects
def get_lines(paragraph, trunc_row):
	"""Instead of array of images, return array of Line objects
	to maintain coordinates of each line in the original image"""
	img = paragraph.img
	lines = []
	l = len(trunc_row)
	if l < 1: return []
	height = []
	for i in range (0, l):
		height.append(trunc_row[i][1] - trunc_row[i][0]);

	med_height = statistics.median(height)
	for i in range (0, l):
		im = trim_line(img[trunc_row[i][0]:trunc_row[i][1]])
		line_rect = get_rect(img.shape, paragraph.rect, (0, trunc_row[i][0], img.shape[1], trunc_row[i][1]))
		# Add padding to english lines
		bordersize = med_height - height[i]
		if bordersize > 0:
			im = cv2.copyMakeBorder(im, top=int(bordersize), bottom=0, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
		lines.append(Line(img=im, rect=line_rect))
	return lines

# Implemented for testing; saves line images by paragraph and line number
def save_line_imgs(paragraph, trunc_row):
	length = len(trunc_row)
	label = label_location(trunc_row)
	lines = get_lines(paragraph, trunc_row)
	for i in range (0, length):
		name = "test/paragraph" + str(label[i][0]) + "line" + str(label[i][1]) + ".png"
		cv2.imwrite(name, lines[i].img)

# Outputs line imgs in the form of following:
# Divide one paragraph into many
# paragraph_n = [line_1, line_2, ...]
# line_n = line img
def output_line_imgs(paragraph):
	sorted = []
	img = paragraph.img
	trunc_row = find_line(img)
	length = len(trunc_row)
	(_, height) = img.shape
	(num_p, label) = label_paragraph(trunc_row, height)
	for i in range (0, num_p):
		sorted.append(Paragraph())
	lines = get_lines(paragraph, trunc_row)
	for i in range (0, length):
		sorted[label[i]].lines.append(lines[i])
	return sorted

# implemented for testing; returns list of paragraph and line #
def label_location(trunc_row, height):
	label, num_p, num_l = [], 0, 0
	length = len(trunc_row)
	for i in range (0, length - 1):
		label.append((num_p, num_l))
		num_l = num_l + 1
		if trunc_row[i + 1][0] - trunc_row[i][1] > height:
			num_p = num_p + 1
			num_l = 0
	label.append((num_p, num_l))
	return label

# implemented for testing; saves line images by paragraph and line number
def save_line_imgs(img):
	trunc_row = find_line(im_bw)
	length = len(trunc_row)
	(_, height) = img.shape
	label = label_location(trunc_row, height)
	lines = get_lines(Paragraph(img), trunc_row)
	for i in range (0, length):
		name = "test/paragraph" + str(label[i][0]) + "line" + str(label[i][1]) + ".png"
		cv2.imwrite(name, lines[i].img)

# If executed directly
if __name__ == '__main__':
	# get img and change it to grayscale
	original_img = cv2.imread('test/line_testing/test_book.png')
	grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

	# change img to black & white
	(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	(x, y) = im_bw.shape
	save_line_imgs(im_bw)
