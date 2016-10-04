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
    chars = [Char(img[0:32, 0+i*32:i*32+32]) for i in range(6)]
    l = Line(chars)
    p = Paragraph([l])
    return [p]