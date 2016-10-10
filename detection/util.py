from enum import Enum

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

# sums up the column pixel value of a given row in an image
def sumup_column(img, row_number):
	sum = 0
	(_, length) = img.shape
	for i in range (0, length):
		sum = sum + img.item(row_number, i)
	return sum / length

# sums up the row pixel value of a given column in an image
def sumup_row(img, col_number):
	sum = 0
	(length, _) = img.shape
	for i in range (0, length):
		sum = sum + img.item(i, col_number)
	return sum / length