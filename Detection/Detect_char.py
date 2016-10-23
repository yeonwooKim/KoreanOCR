import cv2
import numpy as np
from enum import Enum
import Detect_line as dl

original_img = cv2.imread('test/line_testing/test2.png')
grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
(_, im_bw) = cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
(x, y) = im_bw.shape

essay = dl.output_line_imgs(im_bw, dl.find_line(im_bw))

class Paragraph:
    def __init__(self, lines):
        self.lines = lines

class Line:
    def __init__(self, chars):
        self.chars = chars

class CHARTYPE(Enum):
    CHAR = 0
    BLANK = 1

# 각 문자의 정보를 담고 있음
# img는 32 X 32 numpy array여야 함
# blank, 즉 띄어쓰기도 하나의 char로 간주하여 추가해주어야함
class Char:
    def __init__(self, img):
        self.img = img
        self.type = CHARTYPE.CHAR

# reconst 모듈로 넘겨줄 paragraph list를 생성
# 아래 구현은 이미지에서 왼쪽 상단 192 X 64 부분을 잘라
# 12개 char로 만들어 넘겨주는 임시 구현
def get_graphs(img):
    chars = [Char(imresize(img[0:32, 0+i*32:i*32+32], [32, 32])) for i in range(0, 6)]
    l1 = Line(chars)
    chars = [Char(imresize(img[32:64, 0+i*32:i*32+32], [32, 32])) for i in range(0, 6)]
    l2 = Line(chars)
    p = Paragraph([l1, l2])
    return [p]

# Sums up the column pixel value of a given column in an image
def sumup_col(img, col_number):
	sum = 0
	(length, _) = img.shape
	for i in range (0, length):
		sum = sum + img.item(i, col_number)
	return sum / length

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
	max = -1

	(height, length) = line.shape
	for i in range (0, length):
		sum = sumup_col(line, i)
		if sum > 0 and start_letter == -1:
			if i == 0:
				start_letter = i
			else:
				start_letter = i - 1
			if start_letter - end_letter > 5:
				if word != []:
					candidate.append(word)
					word = []
		elif sum > 0 or (sum == 0 and start_letter == -1):
			continue
		elif sum == 0:
			end_letter = i
			width = end_letter - start_letter
			if width > max and height > width:
				max = width
			word.append((start_letter, end_letter))
			start_letter = -1

	if start_letter != -1:
		width = length - 1 - start_letter
		if width > max and height > width:
			max = width
		word.append((start_letter, length - 1))
	if word != []:
		candidate.append(word)
	return (max, candidate)

# Calculates the difference of end and start pts of each letter, returns a list of widths
def calc_width(word):
	width = []
	for i in word:
		width.append(i[1] - i[0])
	return width

# Merges and splits letters considering the context
# Return value:
#		[ word1, word2, word3, ... , wordN ]
# wordN ::= [ letter1, letter2, ... , letterN ]
# letterN ::= ( starting index, ending index )
def snd_pass(line, max, candidate):
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
			if max - 7 <= word[i + 1][1] - word[i][0] < height and width[i] >= width[i + 1]:
				snd_word.append((word[i][0], word[i + 1][1]))
				merged = 1
			elif max - 2 <= width[i] < height:
				snd_word.append(word[i])
			elif width[i] > height:
				# split word; not yet implemented
				snd_word.append(word[i])
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
	snd_candidate = []
	letter_size = -1
	for line in para:
		(m, c) = fst_pass(line)
		fst_candidate.append(c)
		if letter_size < m:
			letter_size = m
	l = len(para)
	for i in range (0, l):
		c = snd_pass(para[i], letter_size, fst_candidate[i])
		snd_candidate.append(c)
	return snd_candidate

# Implemented for testing; truncates line image to letters and saves letter images
# with word and letter information
def trunc_n_save_letter(para_num, line_num, line, candidate):
	(x, y) = line.shape
	cnt_word = 0
	for word in candidate:
		cnt_letter = 0
		for letter in word:
			cv2.imwrite('test/char_testing/p' + str(para_num) + 'l' + str(line_num) +
					'w' + str(cnt_word) + 'l' + str(cnt_letter) + '.png', line[0:x,letter[0]:letter[1]])
			cnt_letter = cnt_letter + 1
		cnt_word = cnt_word + 1

# Implemented for testing; Given an essay, processes 2 times and saves letters as images
def save_essay(essay):
	l = len(essay)
	for i in range (0, l):
		c = proc_paragraph(essay[i])
		l2 = len(essay[i])
		for j in range (0, l2):
			trunc_n_save_letter(i, j, essay[i][j], c[j])

save_essay(essay)
