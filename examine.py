import getopt
import os
import sys
from time import strftime

import cv2
from PIL import Image, ExifTags
import numpy as np

import detection
import reconst
import semantic
import chrecog
from preprocessing import preprocess_image
from detection.util import CHARTYPE, Char, Line, Paragraph

import matplotlib.pyplot as plt

pid = os.getpid()

def print_msg(msg):
    time_str = strftime("%Y-%m-%d %H:%M:%S")
    print ("[%5d] %s %s" % (pid, time_str, msg))

def simple_preproc(img, threshold=True):
    if len(img.shape) > 2:
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_img = img

    if not threshold:
        return grayscale_img

    return cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def get_txt(img, is_simple=False):
    """이미지를 각 모듈에 순서대로 넘겨줌.
    분석된 최종 문자열을 반환"""
    print_msg("preprocessing..")
    if is_simple:
        processed_imgs = [simple_preproc(img)]
    else:
        layouts = preprocess_image(img)
        processed_imgs = []
        for l in layouts:
            if hasattr(l.image, "shape"):
                #plt.figure(num=None, figsize=(3, 3), facecolor='w', edgecolor='k')
                #plt.imshow(l.image, interpolation="none", cmap=plt.get_cmap("gray"))
                #plt.show()
                processed_imgs.append(l.image)

    print_msg("detecting..")
    graphs = []
    for processed in processed_imgs:
        graphs.extend(detection.get_graphs(processed))

    print_msg("recognizing..")
    graphs = chrecog.predict.get_pred(graphs)

    print_msg("semantic..")
    graphs = semantic.analyze(graphs)

    print_msg("reconst..")
    return reconst.build_graphs(graphs)

msg_help = """python examine.py <image_path>
-l --letter letter mode
-i --invert invert image"""

def get_char(img):
    processed = simple_preproc(img)
    chars = [Char([0, processed.shape[1]], CHARTYPE.CHAR)]
    lines = [Line(processed, chars)]
    graphs = [Paragraph(lines)]

    graphs = chrecog.predict.get_pred(graphs)

    graphs = semantic.analyze(graphs)
    return reconst.build_graphs(graphs)

def pil_to_cv(image):
    for orientation in ExifTags.TAGS.keys(): 
        if ExifTags.TAGS[orientation]=='Orientation' : break 
    if image._getexif() is not None:
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def main(argv):
    letter = False
    is_simple = False
    invert = False
    try:
        opts, args = getopt.gnu_getopt(argv, "hli", ["help", "letter", "invert", "sp"])
    except getopt.GetoptError:
        print(msg_help)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(msg_help)
            sys.exit()
        elif opt in ("-l", "--letter"):
            letter = True
        elif opt in ("-i", "--invert"):
            invert = True
        elif opt in ("--sp"):
            is_simple = True

    if len(args) != 1:
        print(msg_help)
        sys.exit(2)

    img = pil_to_cv(Image.open(args[0]))
    if img is None:
        print("Invalid image file")
        exit(1)

    if invert:
        img = 255-img

    if not letter:
        print(get_txt(img, is_simple))
    else:
        print(get_char(img))

if __name__ == "__main__":
    main(sys.argv[1:])
