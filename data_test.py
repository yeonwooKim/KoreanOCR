# data_test.py
# Used to test data generated correctly
# python data_test.py --help

import sys, getopt
from data import en_chset, gen
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("plt import error")
    exit(-1)
import itertools
import random
import math

msg_help = """python data_test.py <font name>
-h (help)
-e English
-k Korean"""

def main(argv):
    font_name = None
    english = False
    korean = False
    try:
        opts, args = getopt.gnu_getopt(argv,"hek",["help"])
    except getopt.GetoptError:
        print(msg_help)
        sys.exit(-1)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(msg_help)
            sys.exit()
        elif opt == "-e":
            english = True
        elif opt == "-k":
            korean = True
            
    if len(args) != 1:
        print(msg_help)
        sys.exit(-1)
    
    font_name = args[0]
    gen.supported_fonts.append(font_name)
    if (english):
        for ch in [".", ",", "j", "g", "p", "q", "y", "i", "《", "》", "『", "「"]:
            show_example(ch, font_name)
    if (korean):
        for ch in ["가", "낢", "뷁", "헿"]:
            show_example(ch, font_name)

# Helper function to draw 3 X 3 plots
def draw_subplot(array, w, h):
    plt.figure(num=None, figsize=(3, 3), facecolor='w', edgecolor='k')
    for i in range(len(array)):
        plt.subplot(w,h,i+1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(array[i], interpolation="none", cmap=plt.get_cmap("gray"))
    
def show_example(ch, font):
    mat = gen.get_mat(ch, font, "NORMAL")
    if (mat is None):
        print("passing %s %s" % (ch, font))
        return
    sliced = [gen.slice_img(mat) for i in range(7)]
    draw_subplot(sliced, 1, 7)
    plt.show()
        
if __name__ == "__main__":
    main(sys.argv[1:])
