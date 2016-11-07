from enum import Enum

class Paragraph:
	def __init__(self, lines):
		self.lines = lines

class Line:
	def __init__(self, img, chars):
		self.chars = chars

class CHARTYPE(Enum):
    CHAR = 0
    BLANK = 1
    def __eq__(self, other):
        return self.value == other.value

# 각 문자의 정보를 담고 있음
# img는 32 X 32 numpy array여야 함
# blank, 즉 띄어쓰기도 하나의 char로 간주하여 추가해주어야함
class Char:
	def __init__(self, pt, type):
		self.pt = pt
		self.type = type
		self.children = []

	def add_child(self, obj):
		self.children.append(obj)

	def get_child(self, pt):
		for child in self.children:
			if child.pt == pt:
				return child
		return None

# Sums up the row pixel value of a given row in an image
def sumup_row(img, row_number):
	sum = 0
	(_, length) = img.shape
	for i in range (0, length):
		sum = sum + img.item(row_number, i)
	return sum / length

# Sums up the column pixel value of a given column in an image
def sumup_col(img, col_number):
	sum = 0
	(length, _) = img.shape
	for i in range (0, length):
		sum = sum + img.item(i, col_number)
	return sum / length

# Trims line images
def trim_line(img):
	(x, y) = img.shape
	fst, lst = -1, -1
	for i in range (0, y):
		sum = sumup_col(img, i)
		if sum > 0:
			fst = i
			break
	for i in range (0, y):
		sum = sumup_col(img, y - 1 - i)
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
