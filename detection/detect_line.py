from __future__ import division
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import statistics
from util import *

FIND_ROW = 0
FIND_COL = 1

#if __name__ == '__main__':
	# Implemented for testing; get img and change it to grayscale
	#original_img = cv2.imread('test/line_testing/test2.png')
	#grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

	# Implemented for testing; change img to black & white
	#(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#(x, y) = im_bw.shape

# Finds the lines that contain text and
# Return them as a list of tuples that indicate the location of starting and ending pt of each line
def find_trunc(img, row_col=FIND_ROW, blank_thd = 2000, distance_thd = 0, min_len = 5):
	start_idx, end_idx = -1, -1
	distance = 0
	trunc = []

	if row_col == FIND_ROW:
		dim = img.shape[0]
	elif row_col == FIND_COL:
		dim = img.shape[1]
	else:
		raise ValueError("Invalid row_col")

	for i in range (dim):
		if row_col == FIND_ROW:
			sum = sumup_row(img, i)
		else:
			sum = sumup_col(img, i)

		if start_idx == -1:
			if sum >= blank_thd:
				start_idx = i
				distance = 0
		else:
			if sum >= blank_thd:
				end_idx = -1
				distance = 0
				continue
			else:
				if end_idx == -1:
					if i - start_idx < min_len and len(trunc) > 0:
						trunc[-1] = (trunc[-1][0], i) # expand last elm
						continue
					end_idx = i
						
				if distance >= distance_thd:
					distance = 0
					trunc.append((start_idx, end_idx))
					start_idx, end_idx = -1, -1
				else:
					distance += 1
			
	
	if start_idx != -1:
		trunc.append((start_idx, dim - 1))

	return trunc

'''
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
'''
# Truncates the paragraph to lines
# Returns list of Line objects
def get_lines(img, rect, row_col = FIND_ROW, blank_thd = 2000, distance_thd = 0):
	"""Instead of array of images, return array of Line objects
	to maintain coordinates of each line in the original image"""
	trunc = find_trunc(img, row_col, blank_thd, distance_thd)
	lines = []
	l = len(trunc)
	if l < 1: return []

	if row_col == FIND_ROW:
		height = []
		for i in range (0, l):
			height.append(trunc[i][1] - trunc[i][0]);

		med_height = statistics.median(height)
		for i in range (0, l):
			#im = trim_line(img[trunc[i][0]:trunc[i][1]])
			# Add padding to english lines
			if med_height > height[i]:
				pad = int(height[i] * 0.1)
				trunc_rect = (0, max(0, trunc[i][0]-pad), img.shape[1], min(img.shape[0], trunc[i][1]+pad))
			else:
				pad = int(height[i] * 0.05)
				trunc_rect = (0, max(0, trunc[i][0]-pad), img.shape[1], min(img.shape[0], trunc[i][1]+pad))

			im = img[trunc_rect[1]:trunc_rect[3]]
			line_rect = get_rect(img.shape, rect, trunc_rect)
			lines.append(Line(img=im, rect=line_rect))
	else:
		for trunc_elm in trunc:
			im = img[:, trunc_elm[0]:trunc_elm[1]]
			line_rect = get_rect(img.shape, rect, (trunc_elm[0], 0, trunc_elm[1], img.shape[0]))
			lines.append(Line(img=im, rect=line_rect))
	return lines
'''
# Implemented for testing; saves line images by paragraph and line number
def save_line_imgs(paragraph, trunc_row):
	length = len(trunc_row)
	label = label_location(trunc_row)
	lines = get_lines(paragraph, trunc_row)
	for i in range (0, length):
		name = "test/paragraph" + str(label[i][0]) + "line" + str(label[i][1]) + ".png"
		cv2.imwrite(name, lines[i].img)
'''
BLANK_THD_FACTOR = 1.5 # 높으면 글자들이 더 잘게 잘림
OUTPUT_LINE_SIZE = 48
def get_paragraph_lines(para_img, para_rect):
	lines_1 = get_lines(para_img, para_rect, FIND_ROW, 1000 * BLANK_THD_FACTOR, 0)
	lines_2 = []
	lines_3 = []

	for line in lines_1:
		if line.img.shape[0] < 6 or line.img.shape[1] < 2:
			continue
		if line.img.shape[1] > line.img.shape[0] * 14:
			distance_thd = line.img.shape[0] * 0.5 # It's likely long line
		else:
			distance_thd = line.img.shape[0] * 0.3
		lines_2.extend(get_lines(line.img, line.rect, FIND_COL, line.img.shape[0] * BLANK_THD_FACTOR, distance_thd))

	for line in lines_2:
		if line.img.shape[0] < 6 or line.img.shape[1] < 2:
			continue
		# 가로가 세로보다 길다면, row를 더 많이 자른다
		if line.img.shape[1] > line.img.shape[0] * 1.1:
			distance_thd = 0
		else:
			distance_thd = line.img.shape[0] * 0.3
		lines_3.extend(get_lines(line.img, line.rect, FIND_ROW, line.img.shape[1] * BLANK_THD_FACTOR, distance_thd))

	ret = []
	for line in lines_3:
		if line.img.shape[0] < 6 or line.img.shape[1] < 2:
			continue
		ret.append(line)

	#for line in lines_3:
	#	if line.img.shape[0] > OUTPUT_LINE_SIZE:
	#		scale = 1.0 * OUTPUT_LINE_SIZE / line.img.shape[0]
	#		line.img = cv2.resize(line.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

	return ret

# Outputs line imgs in the form of following:
# Divide one paragraph into many
# paragraph_n = [line_1, line_2, ...]
# line_n = line img
'''
def output_line_imgs(paragraph):
	sorted = []
	img = paragraph.img
	trunc_row = find_row(img)
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
	trunc_row = find_row(im_bw)
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
'''