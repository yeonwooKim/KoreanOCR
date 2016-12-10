# data_gen.py
# Used to generate composite data
# python data_gen.py --help

import getopt
import itertools
import math
import random
import sys

import matplotlib

from data import en_chset, gen

plot_disabled = False
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("plt import error : plot disabled")
    plot_disabled = True

msg_help = """python data_gen.py <save path>
-h (help)
-f (do not ask to proceed)
-r (refresh system font cache)
-s <number of data>
--noplot (do not show plot)"""

f = open("data/k1001.txt")
ko_chset = f.read().splitlines()

all_chset = ko_chset + en_chset

def main(argv):
    save_path = None
    datasize = 300000
    force = False
    refresh = False
    plot = True
    try:
        opts, args = getopt.gnu_getopt(argv,"hfrs:",["help","force","refresh","size=","noplot"])
    except getopt.GetoptError:
        print(msg_help)
        sys.exit(-1)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(msg_help)
            sys.exit()
        elif opt in ("-f", "--force"):
            force = True
        elif opt in ("-s", "--size"):
            datasize = int(arg)
        elif opt in ("-r", "--refresh"):
            refresh = True
        elif opt == "--noplot":
            plot = False
            
    if len(args) != 1 or datasize <= 0:
        print(msg_help)
        sys.exit(-1)

    if refresh:
        print("Refreshing font cache...")
        gen.refresh_font_cache()
    
    save_path = args[0]
    generate_fonts(save_path, datasize, plot and not plot_disabled, force)

def get_random_ch(chset=all_chset):
    return chset[random.randrange(0,len(chset))]

# Helper function to draw 3 X 3 plots
def draw_subplot(array, w, h):
    plt.figure(num=None, figsize=(3, 3), facecolor='w', edgecolor='k')
    for i in range(len(array)):
        plt.subplot(w,h,i+1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(array[i], interpolation="none", cmap=plt.get_cmap("gray"))

def show_example_all(ch, fonts, weights):
    sliced = []
    for font, weight in itertools.product(fonts, weights):
        mat = gen.get_mat(ch, font, weight)
        if mat is None:
            print("Skip %s for %s %s" % (ch, font, weight))
            continue
        sliced.append(gen.process_mat(mat))
    draw_subplot(sliced, math.ceil(len(sliced)/2), 2)

def show_example(ch):
    sliced = [gen.process_mat(gen.get_mat(ch)) for i in range(7)]
    draw_subplot(sliced, 1, 7)

def show_example_random():
    sliced = [gen.process_mat(gen.get_mat(get_random_ch())) for i in range(7)]
    draw_subplot(sliced, 1, 7)

#for ch in ".", ";", "2", ")", "가", "낢", "·", "《", "》", "「", "」", "『", "』" :
#    show_example(ch)
#
#for i in range(3):
#    show_example_random()

def generate_fonts(save_path, datasize, plot, force=False):
    fonts = gen.supported_fonts[:]
    weights = gen.supported_weights[:]
    unavailable = gen.get_unavailable(fonts)
    if len(unavailable):
        for s in unavailable:
            print("%s not found" % s)
            fonts.remove(s)
        print("========")
        print("Please provide above fonts. After you install new ones,")
        print("make sure you run fc-cache -f -v or run with option -r,")
        print("which would do fc-cache for you.")
    print("==Target fonts==")
    for font in fonts:
        print(font)
    if plot:
        show_example_all("A", fonts, weights)
        plt.show()
    print("================")
    if (force or input("Are you sure to proceed? (y/n)") == "y"):
        print("Proceeding..")
        gen.save_chset_random(fonts, weights, (en_chset, ko_chset), (10, 30, 1), save_path, datasize)
    else:
        print("Aborting..")
        
if __name__ == "__main__":
    main(sys.argv[1:])
