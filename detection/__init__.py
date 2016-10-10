import numpy as np
from scipy.misc import imresize
from detection.detect_char import get_letters_from_line
from detection.detect_line import output_line_imgs
from detection.util import *

def reshape_with_margin(img, size=32, pad=1):
    if img.shape[0] > img.shape[1] :
        dim = img.shape[0]
        margin = (dim - img.shape[1])//2
        margin_img = np.zeros([dim, margin])
        reshaped = np.c_[margin_img, img, margin_img]
    else :
        dim = img.shape[1]
        margin = (dim - img.shape[0])//2
        margin_img = np.zeros([margin, dim])
        reshaped = np.r_[margin_img, img, margin_img]
    reshaped = imresize(reshaped, [size-pad*2, size-pad*2])
    padded = np.zeros([size, size])
    padded[pad:-pad, pad:-pad] = reshaped
    return padded

# return [Paragraph]
def get_graphs(img):
    essay = output_line_imgs(img)
    ret = []
    for lineimgs in essay:
        p = Paragraph([])
        for lineimg in lineimgs:
            charimgs = get_letters_from_line(lineimg)
            chars = []
            for charimg in charimgs:
                chars.append(Char(reshape_with_margin(charimg)))
            l = Line(chars)
            p.lines.append(l)
        ret.append(p)
    return ret