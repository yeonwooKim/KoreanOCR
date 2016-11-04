import cv2
import numpy as np
from enum import Enum
import detect_line as dl
import statistics
import math
import util

original_img = cv2.imread('test/line_testing/test3.png')
grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
(x, y) = im_bw.shape

essay = dl.output_line_imgs(im_bw)

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
	
	blank_thd = 10

	(height, length) = line.shape
	for i in range (0, length):
		sum = util.sumup_col(line, i)
		if sum > blank_thd and start_letter == -1:
			if i == 0:
				start_letter = i
			else:
				start_letter = i - 1
			if start_letter - end_letter > 0.5 * height:
				if word != []:
					candidate.append(word)
					word = []
		elif sum > blank_thd or (sum <= blank_thd and start_letter == -1):
			continue
		elif sum <= blank_thd:
			end_letter = i
			width = end_letter - start_letter
			wlist.append(width)
			word.append((start_letter, end_letter))
			start_letter = -1

	if start_letter != -1:
		width = length - 1 - start_letter
		wlist.append(width)
		word.append((start_letter, length - 1))
	if word != []:
		candidate.append(word)
	return (statistics.median(wlist), candidate)

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
		pt, m = -1, math.inf
		for j in range (i, w):
			s = util.sumup_col(img, j)
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
def snd_pass(line, size, candidate, merge_thdl, merge_thdh):
	snd_candidate = []
	(height, _) = line.shape
	for word in candidate:
		snd_word = []
		width = calc_width(word)
		length = len(width)
		merged = 0
		for i in range (0, length - 1):
			if merged == 1:
				merged = 0
				continue
			if size * merge_thdl <= word[i + 1][1] - word[i][0] <= height * merge_thdh and width[i] >= width[i + 1]:
				snd_word.append((word[i][0], word[i + 1][1]))
				merged = 1
			elif width[i] > height:
				num_letter = width[i] // size
				pts = find_split_pts(word[i][0], num_letter, size, line[0:, word[i][0]:word[i][1]])
				l = len(pts)
				for j in range (0, l):
					snd_word.append(pts[j])
					if j == l - 1:
						w = word[i][1] - pts[j][1]
						if (size * merge_thdl) <= word[i + 1][1] - pts[j][1] <= (height * merge_thdh) and w >= width[i + 1]:
							snd_word.append((pts[j][1], word[i + 1][1]))	
							merged = 1
						else:
							snd_word.append((pts[j][1], word[i][1]))
			else:
				snd_word.append(word[i])
		if merged == 0:
			snd_word.append(word[length - 1])
		snd_candidate.append(snd_word)
		snd_word = []
	return snd_candidate

# Given a paragraph, processes 2 times and returns the candidate word points
def proc_paragraph(para):
	fst_candidate = []
	snd_candidate1 = []
	snd_candidate2 = []
	snd_candidate3 = []
	snd_candidate4 = []
	wlist = []
	for line in para:
		(m, c) = fst_pass(line)
		fst_candidate.append(c)
		wlist.append(m)
	l = len(para)
	letter_size = statistics.median(wlist)
	for i in range (0, l):
		c = snd_pass(para[i], letter_size, fst_candidate[i], 1.1, 0.9)
		snd_candidate1.append(c)
		c = snd_pass(para[i], letter_size, fst_candidate[i], 1.1, 1.1)
		snd_candidate2.append(c)
		c = snd_pass(para[i], letter_size, fst_candidate[i], 0.9, 0.9)
		snd_candidate3.append(c)
		c = snd_pass(para[i], letter_size, fst_candidate[i], 1.1, 0.9)
		snd_candidate4.append(c)
	return (snd_candidate1, snd_candidate2, snd_candidate3, snd_candidate4)

# Implemented for testing; truncates line image to letters and saves letter images
# with word and letter information
def trunc_n_save_letter(para_num, line_num, pass_num, line, candidate):
	(x, y) = line.shape
	cnt_word = 0
	for word in candidate:
		cnt_letter = 0
		for letter in word:
			cv2.imwrite('test/char_testing/t' + str(pass_num) + '/p' + str(para_num) + 'l' + str(line_num) +
					'w' + str(cnt_word) + 'l' + str(cnt_letter) + '.png', line[0:x,letter[0]:letter[1]])
			cnt_letter = cnt_letter + 1
		cnt_word = cnt_word + 1

# Implemented for testing; Given an essay, processes 2 times and saves letters as images
def save_essay(essay):
	l = len(essay)
	for i in range (0, l):
		(c1, c2, c3, c4) = proc_paragraph(essay[i])
		l2 = len(essay[i])
		for j in range (0, l2):
			trunc_n_save_letter(i, j, 1, essay[i][j], c1[j])
			trunc_n_save_letter(i, j, 2, essay[i][j], c2[j])
			trunc_n_save_letter(i, j, 3, essay[i][j], c2[j])
			trunc_n_save_letter(i, j, 4, essay[i][j], c2[j])

save_essay(essay)

# Truncates line image to letters and constructs line classes
# Resizes char images to 32 * 32
def to_line(line, candidate):
	l = []
	length = len(candidate)
	for i in range (0, length):
		for letter in candidate[i]:
			char = Char(cv2.resize(line[0:,letter[0]:letter[1]], (32, 32)), CHARTYPE.CHAR)
			l.append(char)
		if i != length - 1:
			l.append(Char(None, CHARTYPE.BLANK))
	return Line(l)

# Given a paragraph image, constructs a paragraph class
def to_paragraph(para):
	l = []
	c = proc_paragraph(para)
	length = len(para)
	for i in range (0, length):
		line = to_line(para[i], c[i])	
		l.append(line)
	return Paragraph(l)

# reconst 모듈로 넘겨줄 paragraph list를 생성
def get_graphs(img):
	essay = dl.output_line_imgs(img, dl.find_line(im_bw))
	l = []
	length = len(essay)
	for i in range (0, length):
		para = to_paragraph(essay[i])
		l.append(para)
	return l
