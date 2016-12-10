import os, sys, getopt
import numpy as np
import cv2

import preproc
import detection
from detection.util import Paragraph, Line, Char, CHARTYPE
from chrecog.predict import get_session, load_ckpt, get_pred
import reconst
import semantic
from time import strftime

pid = os.getpid()

def print_msg(msg):
    time_str = strftime("%Y-%m-%d %H:%M:%S")
    print ("[%5d] %s %s" % (pid, time_str, msg))

# 이미지를 각 모듈에 순서대로 넘겨줌
# 분석된 최종 문자열을 반환
def get_txt(img):
    print_msg("preprocessing..")
    processed_imgs = [preproc.process(img)]

    print_msg("detecting..")
    graphs = []
    for processed in processed_imgs:
        graphs.extend(detection.get_graphs(processed))

    print_msg("recognizing..")
    sess = get_session()
    load_ckpt(sess, "data/ckpt/161117_BN2.ckpt")
    graphs = get_pred(graphs)

    print_msg("semantic..")
    graphs = semantic.analyze(graphs)

    print_msg("reconst..")
    return reconst.build_graphs(graphs)

msg_help = """python examine.py <image_path>"""

def get_char(img):
    processed = preproc.process(img)
    chars = [Char([0, processed.shape[1]], CHARTYPE.CHAR)]
    lines = [Line(processed, chars)]
    graphs = [Paragraph(lines)]

    sess = get_session()
    load_ckpt(sess, "data/ckpt/161117_BN2.ckpt")
    graphs = get_pred(graphs)

    graphs = semantic.analyze(graphs)
    return reconst.build_graphs(graphs)


def main(argv):
    letter = False
    try:
        opts, args = getopt.gnu_getopt(argv, "hl", ["help", "letter"])
    except getopt.GetoptError:
        print(msg_help)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(msg_help)
            sys.exit()
        elif opt in ("-l", "--letter"):
            letter = True

    if len(args) != 1:
        print(msg_help)
        sys.exit(2)

    img = cv2.imread(args[0])
    if img is None:
        print("Invalid image file")
        exit(1)

    if not letter:
        print(get_txt(img))
    else:
        print(get_char(img))

if __name__ == "__main__":
    main(sys.argv[1:])
