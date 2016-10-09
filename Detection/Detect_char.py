import cv2
import numpy as np
from enum import Enum

original_img = cv2.imread('test/paragraph0line4.png')
grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

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

# sums up the row pixel value of a given column in an image
def sumup_row(img, col_number):
	sum = 0
	(length, _) = img.shape
	for i in range (0, length):
		sum = sum + img.item(i, col_number)
	return sum / length

# returns all the candidate letter points as a list of pairs
def fst_pass(line):
	candidate = []
	word = []
	start_letter, end_letter = -1, -1
	(_, length) = line.shape
	for i in range (0, length):
		sum = sumup_row(line, i)
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
			word.append((start_letter, end_letter))
			start_letter = -1
	return candidate

# implemented for testing; truncates line image to letters and saves letter images
# with word and letter information
def trunc_n_save_letter(line, candidate):
	(x, y) = line.shape
	cnt_word = 0
	for word in candidate:
		cnt_letter = 0
		for letter in word:
			cv2.imwrite('word' + str(cnt_word) + 'letter' + str(cnt_letter) + '.png', line[0:x,letter[0]:letter[1]])
			cnt_letter = cnt_letter + 1
		cnt_word = cnt_word + 1

trunc_n_save_letter(grayscale_img, fst_pass(grayscale_img))

