# detect.py
# Layout detection, line detection, character detection
from scipy.misc import imresize
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