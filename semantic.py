#import matplotlib.pyplot as plt
#plt.rcParams['image.cmap'] = 'Greys'
from detection.util import CHARTYPE
import numpy as np

def print_recur(i_l, indent, clist, force_print=False):
    for c in clist:
        print("%s (%2d, %4d, %4d)" % (' ' * indent, i_l, c.pt[0], c.pt[1]), end=" ")
        if hasattr(c, "prob"):
            if c.value is None:
                print("%.3f %s" % (c.prob, "invalid"))
            else:
                print("%.3f %s" % (c.prob, c.value))
        else:
            print("")
        if len(c.children) > 0:
            print_recur(i_l, indent + 4, c.children, True)

def analyze_recur(clist):
    for c in clist:
        if c.type == CHARTYPE.BLANK:
            c.pt = [-1, -1]
            c.prob, c.value = 1, " "
        elif hasattr(c, "pred"):
            c.prob, c.value = c.pred.sure, c.pred.candidate
                
        if len(c.children) > 0:
            analyze_recur(c.children)
        
# Take candidate with lowest 'rotten point'
# rotten: number of invalid symbols
# tail: length of symbols - 1
# prob: lowest probability of symbols
def merge_children(clist):
    for c in clist:
        cand = None
        merge_children(c.children)
        for i, child in enumerate(c.children):
            if cand is None:
                cand = child
            elif child.rotten_point < cand.rotten_point:
                cand = child
        if cand is None:
            c.tail = 0
            if c.value is None:
                c.value = "?"
                c.rotten = 1
            else:
                c.rotten = 0
        else:
            c.tail = cand.tail + 1
            if hasattr(c, "prob"):
                c.prob = min(c.prob, cand.prob)
                if c.value is None:
                    c.value = "?" + cand.value
                    c.rotten = cand.rotten + 1
                else:
                    c.value += cand.value
                    c.rotten = cand.rotten
            else:
                c.prob = cand.prob
                c.value = cand.value
                c.rotten = cand.rotten
        c.rotten_point = c.rotten * 50 + (1-c.prob) * 100 + c.tail

def analyze(graphs):
    len_l = 0
    for p in graphs:
        for l in p.lines:
            len_l += 1
    #plt.figure()
    i_l = 0
    for p in graphs:
        for l in p.lines:
            prev = None
            #plt.subplot(len_l,1, i_l+1)
            #plt.imshow(l.img)
            analyze_recur(l.chars)
            print_recur(i_l, 0, l.chars, True)
            merge_children(l.chars)
            #for c in l.chars:
            #    print("(%2d, %4d, %4d, %4d) %s" % (i_l, c.pt[0], c.pt[1], c.rotten_point, c.value))
            #analyze_pedigree(l.chars)
            #analyze_linear(l.chars)
            i_l+=1
    #plt.show()
    return graphs
