from scipy.misc import imresize

class Paragraph:
    def __init__(self, lines):
        self.lines = lines

class Line:
    def __init__(self, chars):
        self.chars = chars

class Char:
    def __init__(self, img):
        self.img = img
        self.type = "char"

def get_graphs(img):
    chars = [Char(imresize(img[0:32, 0+i*32:i*32+32], [32, 32])) for i in range(0, 6)]
    l1 = Line(chars)
    chars = [Char(imresize(img[32:64, 0+i*32:i*32+32], [32, 32])) for i in range(0, 6)]
    l2 = Line(chars)
    p = Paragraph([l1, l2])
    return [p]