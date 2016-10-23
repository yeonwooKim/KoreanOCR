import numpy as np
from scipy.misc import imresize
from detection.detect_char import get_letters_from_line
from detection.detect_line import output_line_imgs
from detection.util import *

# return [Paragraph]
def get_graphs(img):
    essay = output_line_imgs(img)
    ret = []
    for lineimgs in essay:
        p = Paragraph([])
        for lineimg in lineimgs:
            p.lines.append(Line(get_letters_from_line(lineimg)))
        ret.append(p)
    return ret