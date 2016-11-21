#import matplotlib.pyplot as plt
#plt.rcParams['image.cmap'] = 'Greys'
from detection.util import CHARTYPE
from chrecog.predict import get_pred_one, reshape_with_margin
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

# Analyze tail point which is the measure of too many splits.
# Normally length - 1
def eval_tail(cand):
    point = 0
    # Please please split the . and , symbols...!!
    if cand.value == '.' or cand.value == ',':
        point += 30
    return cand.tail + 1 - point
        

def try_split_rotten(char):
    lpred = get_pred_one(reshape_with_margin(char.img[:, :16]))
    rpred = get_pred_one(reshape_with_margin(char.img[:, 16:]))
    if lpred.candidate is None or rpred.candidate is None:
        return False
    if lpred.sure < 0.9 or rpred.sure < 0.9:
        return False
    char.prob = min((lpred.sure, rpred.sure))
    char.value = lpred.candidate + rpred.candidate
    return True

# Take candidate with lowest 'rotten point'
# rotten: number of invalid symbols
# tail: refer to eval_tail
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
            if c.value is None and not try_split_rotten(c):
                c.value = "?"
                c.rotten = 1
            else:
                c.rotten = 0
        else:
            c.tail = eval_tail(cand)
            if hasattr(c, "prob"):
                c.prob = min(c.prob, cand.prob)
                if c.value is None and not try_split_rotten(c):
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

def analyze_sibiling(clist):
    prev = None
    for child in clist:
        if child.value is None:
            prev = None
            continue
        cl = list(child.value)
        for i, c in enumerate(cl):
            if prev is None:
                prev = c
                continue
            if prev == '다' and c == '·':
                cl[i] = '.'
            prev = c
        child.value = "".join(cl)

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
            #print_recur(i_l, 0, l.chars, True)
            merge_children(l.chars)
            analyze_sibiling(l.chars)
            #for c in l.chars:
            #    print("(%2d, %4d, %4d, %4d) %s" % (i_l, c.pt[0], c.pt[1], c.rotten_point, c.value))
            i_l+=1
    #plt.show()
    return graphs
