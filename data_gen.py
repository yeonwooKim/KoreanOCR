import sys, getopt
from data import en_chset
from data.gen import get_mat, slice_img, save_chset_random, get_unavailable, supported_fonts, supported_weights
import matplotlib.pyplot as plt
import itertools
import random
import math

msg_help = """python data_gen.py <save path>
-h (help)
-f (do not ask)
-s <number of data>
--noplot (do not show plot)"""

f = open("data/k1001.txt")
ko_chset = f.read().splitlines()

all_chset = ko_chset + en_chset

def main(argv):
    save_path = None
    datasize = 300000
    force = False
    plot = True
    try:
        opts, args = getopt.gnu_getopt(argv,"hfs:",["help","force","size=","noplot"])
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
        elif opt == "--noplot":
            plot = False
            
    if len(args) != 1 or datasize <= 0:
        print(msg_help)
        sys.exit(-1)
    
    save_path = args[0]
    generate_fonts(save_path, datasize, plot, force)

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
        sliced.append(slice_img(get_mat(ch, font, weight)))
    draw_subplot(sliced, math.ceil(len(sliced)/2), 2)
    
def show_example(ch):
    sliced = [slice_img(get_mat(ch)) for i in range(7)]
    draw_subplot(sliced, 1, 7)
    
def show_example_random():
    sliced = [slice_img(get_mat(get_random_ch())) for i in range(7)]
    draw_subplot(sliced, 1, 7)
    
#for ch in ".", ";", "2", ")", "가", "낢", "·", "《", "》", "「", "」", "『", "』" :
#    show_example(ch)
#    
#for i in range(3):
#    show_example_random()

def generate_fonts(save_path, datasize, plot, force=False):
    fonts = supported_fonts[:]
    weights = supported_weights[:]
    unavailable = get_unavailable(fonts)
    if (len(unavailable)):
        for str in unavailable:
            print("%s not found" % str)
            fonts.remove(str)
        print("Please provide above fonts.")
    print("==Target fonts==")
    for font in fonts:
        print(font)
    if plot:
        show_example_all("낡", fonts, weights)
    print("================")
    if (force or input("Are you sure to proceed? (y/n)") == "y"):
        print("Proceeding..")
        save_chset_random(fonts, weights, (en_chset, ko_chset), (1, 3), save_path, datasize)
    else:
        print("Aborting..")
        
if __name__ == "__main__":
    main(sys.argv[1:])
