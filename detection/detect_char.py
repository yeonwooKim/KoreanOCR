import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from enum import Enum
import detect_line as dl
import statistics
import math
from util import *

# If executed directly
def main():
	original_img = cv2.imread('../preprocessing/book_sample_0_crop_0.png')
	grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	(x, y) = im_bw.shape

	para = Paragraph(img=im_bw)
	essay = dl.output_line_imgs(para)
	save_essay(essay)
	get_graphs(para)

# Guesses one letter size
def guess_letter_size(wlist, height):
	wlist.sort()
	l = int(height * 0.8)
	h = int(height * 1.2)
	bucket = [0]*(h - l)

	length = len(wlist)
	if length == 0: return 0
	err = 2
	j = 0
	for i in range (l, h):
		if (wlist[j] > i + err):
			break
		flag = 0
		for k in range (j, length):
			if wlist[k] > i - err and flag == 0:
				j = k
				flag = 1
	
			if wlist[k] >= i - err and wlist[k] <= i + err:
				bucket[i - l] = bucket[i - l] + 1
			elif wlist[k] < i - err:
				continue
			else:
				break
	m, index = -1, -1
	for i in range (l, h):
		if (m < bucket[i - l]):
			m, index = bucket[i - l], i

	return index

# Returns all the candidate letter points as a list of pairs and max width of letter
# Return value:
#		( maximum width of a letter in the line,
#				[ word1, word2, word3, ... , wordN ] )
# wordN ::= [ letter1, letter2, ... , letterN ]
# letterN ::= ( starting index, ending index )
def fst_pass(line):
	candidate = []
	word = []
	start_letter, end_letter = -1, -1
	wlist = []
	
	blank_thd = 5100 # Fix this to adjust blank column threshold

	(height, length) = line.shape
	for i in range (0, length):
		sum = sumup_col(line, i)
		if sum > blank_thd and start_letter == -1:
			if i == 0:
				start_letter = i
			else:
				start_letter = i - 1
			if start_letter - end_letter > 0.2 * height: # Fix this to adjust space threshold
				if word != []:
					candidate.append(word)
					word = []
		elif sum > blank_thd or (sum <= blank_thd and start_letter == -1):
			continue
		elif sum <= blank_thd:
			end_letter = i
			width = end_letter - start_letter
			if (width >= 4): # Fix this to adjust noise threshold
				wlist.append(width)
				word.append((start_letter, end_letter))
			start_letter = -1

	if start_letter != -1:
		width = length - 1 - start_letter
		wlist.append(width)
		word.append((start_letter, length - 1))
	if word != []:
		candidate.append(word)
	return (guess_letter_size(wlist, height), candidate)

# Calculates the difference of end and start pts of each letter, returns a list of widths
def calc_width(word):
	width = []
	for i in word:
		width.append(i[1] - i[0])
	return width

def find_split_pts(index, num_letter, size, img):
	(_, w) = img.shape
	i = 0
	pts = []
	while num_letter != 0:
		pt, m = -1, float("inf")
		for j in range (i + round(size) + 1, min(w, i + round(size) + 5)):
			s = sumup_col(img, j)
			if (s <= m):
				pt = j
				m = s
		if pt == -1:
			pt = w - 1
		pts.append((index + i, index + pt))
		i = pt
		num_letter = num_letter - 1
	return pts

# Merges and splits letters considering the context
# Return value:
#		[ word1, word2, word3, ... , wordN ]
# wordN ::= [ letter1, letter2, ... , letterN ]
# letterN ::= ( starting index, ending index )
def snd_pass(line, size, candidate, merge_thdl, merge_thdh, flag):
	snd_candidate = []
	(height, _) = line.shape
	for w in candidate:
		word = list(w)
		snd_word = []
		width = calc_width(word)
		length = len(width)
		for i in range (0, length - 1):
			if flag == 0:
				if size * merge_thdl <= word[i + 1][1] - word[i][0] <= height * merge_thdh and width[i] >= width[i + 1]:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				elif width[i] >= height and width[i + 1] < size / 2:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				elif width[i] < size * 0.7 and width[i + 1] > height:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				elif width[i + 1] > height:
					word[i + 1] = (word[i][0], word[i + 1][1])
					width[i + 1] = word[i + 1][1] - word[i][0]
				else:
					snd_word.append(word[i])
			else:
				word[i + 1] = (word[i][0], word[i + 1][1])
				width[i + 1] = word[i + 1][1] - word[i][0]
		snd_word.append(word[length - 1])
		snd_candidate.append(snd_word)
	return snd_candidate

# split merged letters
def thd_pass(line, size, candidate):
	thd_candidate = []
	(height, _) = line.shape
	for word in candidate:
		thd_word = []
		width = calc_width(word)
		length = len(width)
		for i in range (0, length):
			if width[i] > height:
				num_letter = round(width[i] / size)
				pts = find_split_pts(word[i][0], num_letter, size, line[0:,word[i][0]:word[i][1]])
				l = len(pts)
				for j in range (0, l):
					thd_word.append(pts[j])
				if l != 0:
					thd_word.append((pts[l - 1][1], word[i][1]))
				else:
					thd_word.append(word[i])
			else:
				thd_word.append(word[i])
		thd_candidate.append(thd_word)
	return thd_candidate

# Given a paragraph, processes 2 times and returns the candidate word points
def proc_paragraph(para):
	fst_candidate = []
	snd_candidate1 = []
	snd_candidate2 = []
	snd_candidate3 = []
	snd_candidate4 = []
	wlist = []
	for line in para.lines:
		(m, c) = fst_pass(line.img)
		fst_candidate.append(c)
		wlist.append(m)
	if len(wlist) == 0:
		return None
	letter_size = statistics.median(wlist)
	if letter_size < 1:
		return None
	for i, line in enumerate(para.lines):
		c1 = snd_pass(line.img, letter_size, fst_candidate[i], 0.6, 1.3, 0)
		c1 = thd_pass(line.img, letter_size, c1)
		snd_candidate1.append(c1)
		c2 = snd_pass(line.img, letter_size, fst_candidate[i], 0.8, 1.1, 0)
		c2 = thd_pass(line.img, letter_size, c2)
		snd_candidate2.append(c2)
		c3 = snd_pass(line.img, letter_size, fst_candidate[i], 0.7, 0.9, 0)
		c3 = thd_pass(line.img, letter_size, c3)
		snd_candidate3.append(c3)
		# Generous merge threshold
		c4 = snd_pass(line.img, letter_size, fst_candidate[i], None, None, 1)
		c4 = thd_pass(line.img, letter_size, c4)
		snd_candidate4.append(c4)
	return (snd_candidate1, snd_candidate2, snd_candidate3, snd_candidate4)

# Implemented for testing; truncates line image to letters and saves letter images
# with word and letter information
def trunc_n_save_letter(para_num, line_num, pass_num, line, candidate):
	(x, y) = line.shape
	cnt_word = 0
	for word in candidate:
		cnt_letter = 0
		for letter in word:
			filepath = ('test/char_testing/t' + str(pass_num) + '/p' + str(para_num) + 'l' + str(line_num) +
					'w' + str(cnt_word) + 'l' + str(cnt_letter) + '.png')
			success = cv2.imwrite(filepath, line[0:x,letter[0]:letter[1]])
			if not success: print("save %s failed: check if folder exists" % filepath)
			cnt_letter = cnt_letter + 1
		cnt_word = cnt_word + 1

# Implemented for testing; Given an essay, processes 2 times and saves letters as images
def save_essay(essay):
	for i, graph in enumerate(essay):
		(c1, c2, c3, c4) = proc_paragraph(graph)
		for j, line in enumerate(graph.lines):
			trunc_n_save_letter(i, j, 1, line.img, c1[j])
			trunc_n_save_letter(i, j, 2, line.img, c2[j])
			trunc_n_save_letter(i, j, 3, line.img, c3[j])
			trunc_n_save_letter(i, j, 4, line.img, c4[j])

# Given four lists (candidate lists of different threshold),
# finds the indices for each list with the same end points
# returns them in a tuple along with the value of the end point
def find_common_pt(list1, list2, list3, list4):
	i1, i2, i3, i4 = 0, 0, 0, 0
	l1, l2, l3, l4 = len(list1), len(list2), len(list3), len(list4)
	while (i1 < l1 and i2 < l2 and i3 < l3 and i4 < l4):
		e1, e2, e3, e4 = list1[i1][1], list2[i2][1], list3[i3][1], list4[i4][1]
		m = min(e1, e2, e3, e4)
		if e1 == e2 and e2 == e3 and e3 == e4:
			break;
		if m == e1:
			i1 = i1 + 1
		if m == e2:
			i2 = i2 + 1
		if m == e3:
			i3 = i3 + 1
		if m == e4:
			i4 = i4 + 1
	return (m, i1, i2, i3, i4)

# Given the four candidates, each candidate having form of
# candidate ::= [ point1, point2, ... , pointN ]
# pointn ::= (start index, end index)
# returns list of characters
def compare_pass(candidates):
	char_list = []
	(c1, c2, c3, c4) = candidates
	i1, i2, i3, i4 = 0, 0, 0, 0
	l1, l2, l3, l4 = len(c1), len(c2), len(c3), len(c4)
	while (i1 < l1 and i2 < l2 and i3 < l3 and i4 < l4):
		(m, p1, p2, p3, p4) = find_common_pt(c1[i1:], c2[i2:], c3[i3:], c4[i4:])
		char = Char((c1[i1][0], m), CHARTYPE.CHAR)
		if not (p1 == 0 and p2 == 0 and p3 == 0 and p4 == 0):
			curr = char
			for i in range (0, p1 + 1):
				a = Char(c1[i1 + i], CHARTYPE.CHAR)
				curr.add_child(a)
				curr = a
			curr = char
			for i in range (0, p2 + 1):
				child = curr.get_child(c2[i2 + i])
				if child == None:
					a = Char(c2[i2 + i], CHARTYPE.CHAR)
					curr.add_child(a)
					curr = a
				else:
					curr = child
			curr = char
			for i in range (0, p3 + 1):
				child = curr.get_child(c3[i3 + i])
				if child == None:
					a = Char(c3[i3 + i], CHARTYPE.CHAR)
					curr.add_child(a)
					curr = a
				else:
					curr = child
			curr = char
			for i in range (0, p4 + 1):
				child = curr.get_child(c4[i4 + i])
				if child == None:
					a = Char(c4[i4 + i], CHARTYPE.CHAR)
					curr.add_child(a)
					curr = a
				else:
					curr = child
		char_list.append(char)
		i1 = i1 + p1 + 1
		i2 = i2 + p2 + 1
		i3 = i3 + p3 + 1
		i4 = i4 + p4 + 1
	return char_list

def argmin(items):
	min_idxs = []
	min_elm = float("inf")
	for idx, elm in enumerate(items):
		if min_elm > elm:
			min_idxs = [idx]
			min_elm = elm
		elif min_elm == elm:
			min_idxs.append(idx)
	return min_idxs, min_elm

# Truncates line image to letters and constructs line classes
def update_line(line, candidates):
	(c1, c2, c3, c4) = candidates
	for i in range(len(c1)):
		char_list = compare_pass((c1[i], c2[i], c3[i], c4[i]))
		line.chars.extend(char_list)
		if i != len(c1): # If not last word
			line.chars.append(Char(None, CHARTYPE.BLANK))

# Given a paragraph image, constructs a paragraph class
def update_paragraph(para):
	proc = proc_paragraph(para)
	if proc is None: return
	(c1, c2, c3, c4) = proc
	for i, line in enumerate(para.lines):
		update_line(line, (c1[i], c2[i], c3[i], c4[i]))	

# reconst 모듈로 넘겨줄 paragraph list를 생성
# paragraph 하나를 input으로 받아 여러 paragraph로 분리
def get_graphs(paragraph):
	line_graphs = dl.output_line_imgs(paragraph)
	for graph in line_graphs:
		update_paragraph(graph)
	return line_graphs

if __name__ == "__main__":
	main()
