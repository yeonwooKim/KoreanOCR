import numpy as np
from scipy.misc import imresize
import sys, os
import cv2
import concurrent.futures

sys.path.append(os.path.dirname(__file__))

import detect_char, detect_line

OUTPUT_LINE_SIZE = 48
def get_graphs(layouts, threaded=True):
    if not threaded:
        for para in layouts:
            para.lines = detect_line.get_paragraph_lines(para.img, para.rect)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for para in layouts:
                future = executor.submit(detect_line.get_paragraph_lines, para.img, para.rect)
                futures.append(future)
            for para, future in zip(layouts, futures):
                para.lines = future.result()

    for para in layouts:
        for line in para.lines:
            if line.img.shape[0] > OUTPUT_LINE_SIZE:
                scale = 1.0 * OUTPUT_LINE_SIZE / line.img.shape[0]
                line.img = cv2.resize(line.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            line.chars = detect_char.get_char_list(line.img)
    return layouts