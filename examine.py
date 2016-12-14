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
from util import CHARTYPE, Char, Line, Paragraph

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

def pre_reconst(img, verbose=False, is_simple=False):
    if verbose: print_msg("preprocessing..")
    if is_simple:
        layouts = [
            Paragraph(img=simple_preproc(img), rect=(0, 0, img.shape[0], img.shape[1]))
            ]
    else:
        layouts = preprocess_image(img)

    if verbose: print_msg("detecting..")
    graphs = detection.get_graphs(layouts)

    if verbose: print_msg("recognizing..")
    graphs = chrecog.predict.get_pred(graphs)

    if verbose: print_msg("semantic..")
    graphs = semantic.analyze(graphs)
    return graphs

def get_txt(img, verbose=False, is_simple=False):
    """이미지를 각 모듈에 순서대로 넘겨줌.
    분석된 최종 문자열을 반환"""
    graphs = pre_reconst(img, verbose, is_simple)
    if verbose: print_msg("reconst..")
    return reconst.build_graphs(graphs)

def get_json(imgname, img, verbose=False, is_simple=False):
    graphs = pre_reconst(img, verbose, is_simple)
    if verbose: print_msg("reconst..")
    return reconst.build_json(imgname, graphs)

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

def fix_pil_rot(image):
    if image is None: return None

    if not hasattr(image, "_getexif") or image._getexif() is None:
         return image

    for orientation in ExifTags.TAGS.keys(): 
        if ExifTags.TAGS[orientation]=='Orientation' : break 
   
    exif = dict(image._getexif().items())
    
    if orientation not in exif:
        return image

    if exif[orientation] == 3:
        image = image.rotate(180, expand=True)
    elif exif[orientation] == 6:
        image = image.rotate(270, expand=True)
    elif exif[orientation] == 8:
        image = image.rotate(90, expand=True)

    return image

def pil_to_cv(image):
    if image is None: return None
    image = fix_pil_rot(image)
    imgarr = np.array(image)
    if len(imgarr.shape) > 2:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        return cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)

def main(argv):
    letter = False
    is_simple = False
    invert = False
    verbose = False
    is_json = False
    try:
        opts, args = getopt.gnu_getopt(argv, "hlivj", ["help", "letter", "invert", "sp", "verbose", "json"])
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
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-j", "--json"):
            is_json = True
        elif opt in ("--sp"):
            is_simple = True

    if len(args) != 1:
        print(msg_help)
        sys.exit(2)

    try:
        img = pil_to_cv(Image.open(args[0]))
    except:
        print("Invalid image file")
        exit(1)

    if invert:
        img = 255-img

    if letter:
        print(get_char(img))
    elif is_json:
        imgname = os.path.basename(args[0])
        print(get_json(imgname, img, verbose, is_simple))
    else:
        print(get_txt(img, verbose, is_simple))

if __name__ == "__main__":
    main(sys.argv[1:])
