from enum import Enum

class Paragraph:
	def __init__(self, lines=None, img=None, rect=None):
		self.lines = lines
		if self.lines is None: self.lines = []
		self.img = img
		self.rect = rect

class Line:
	def __init__(self, img, chars=None, rect=None):
		self.img = img
		self.chars = chars
		if (self.chars is None): self.chars = []
		self.rect = rect

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
	if length == 0: return 0
	for i in range (0, length):
		sum = sum + img.item(row_number, i)
	return sum / length * 1000

# Sums up the column pixel value of a given column in an image
def sumup_col(img, col_number):
	sum = 0
	(length, _) = img.shape
	if length == 0: return 0
	for i in range (0, length):
		sum = sum + img.item(i, col_number)
	return sum / length * 1000

'''
def sumup_col_outer_weight(img, col_number, weight):
	sum = 0
	(length, _) = img.shape
	if length == 0: return 0
	for i in range (0, length):
		sum = sum + img.item(i, col_number) + (i - length) ** 2 * weight
	return sum / length * 500
'''

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

def get_rect(img_shape, original_rect, crop_rect):
	"""현재 저장하고 있는 이미지의 shape과 그 이미지의 원래 rect,
	그리고 그 이미지를 crop하는 index를 이용해 새 rect를 얻어낸다.
	rect는 가로가 먼저지만 shape는 행이 먼저임에 주의"""
	if original_rect is None: return None
	original_w = original_rect[2] - original_rect[0]
	original_h = original_rect[3] - original_rect[1]
	scale_h, scale_w = original_h / img_shape[0], original_w / img_shape[1]
	crop_l = crop_rect[0] * scale_w + original_rect[0]
	crop_t = crop_rect[1] * scale_h + original_rect[1]
	crop_r = crop_rect[2] * scale_w + original_rect[0]
	crop_b = crop_rect[3] * scale_h + original_rect[1]
	return (crop_l, crop_t, crop_r, crop_b)