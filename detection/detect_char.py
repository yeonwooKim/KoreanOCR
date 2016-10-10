import cv2
import numpy as np
from enum import Enum

# sums up the row pixel value of a given column in an image
def sumup_row(img, col_number):
	sum = 0
	(length, _) = img.shape
	for i in range (0, length):
		sum = sum + img.item(i, col_number)
	return sum / length

# returns all the candidate letter points as a list of pairs and max width of letter
def fst_pass(line):
	candidate = []
	word = []
	start_letter, end_letter = -1, -1
	max = -1

	(height, length) = line.shape
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

# calculates the difference of end and start pts of each letter, returns a list of widths
def calc_width(word):
	width = []
	for i in word:
		width.append(i[1] - i[0])
	return width

# merges and splits letters considering the context
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
			if max - 3 <= width[i] < height - 5:
				snd_word.append(word[i])
			elif max - 7 <= word[i + 1][1] - word[i][0] < height - 3 and width[i] >= width[i + 1]:
				snd_word.append((word[i][0], word[i + 1][1]))
				merged = 1
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

# If executed directly
if __name__ == '__main__':
	original_img = cv2.imread('test/line_testing/paragraph3line1.png')
	grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	trunc_n_save_letter(grayscale_img, snd_pass(grayscale_img, fst_pass(grayscale_img)[0], fst_pass(grayscale_img)[1]))
