#import matplotlib.pyplot as plt
#plt.rcParams['image.cmap'] = 'Greys'
from detection.util import CHARTYPE
from chrecog.predict import get_pred_batch, reshape_with_margin
from hangul_utils import check_syllable, split_syllable_char
import numpy as np

import matplotlib.pyplot as plt

def draw_subplot(array, w, h):
    plt.figure(num=None, figsize=(3, 3), facecolor='w', edgecolor='k')
    for i in range(len(array)):
        plt.subplot(w,h,i+1)
        plt.xticks(())
        plt.yticks(())
        plt.imshow(array[i], interpolation="none", cmap=plt.get_cmap("gray"))
    plt.show()

def print_recur(i_l, indent, clist, force_print=False):
    """Print pretty analysis info"""
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
    """Set character properties according to prediction result"""
    for c in clist:
        if c.type == CHARTYPE.BLANK:
            c.pt = [-1, -1]
            c.prob, c.value = 1, " "
        elif hasattr(c, "pred"):
            c.prob, c.value = c.pred.sure, c.pred.candidate

        if len(c.children) > 0:
            analyze_recur(c.children)

def eval_tail(c, cand):
    """Analyze tail point which is the measure of too many splits.
    They are normally length - 1
    """
    point = 0
    # Please please split the . and , symbols...!!
    if cand.value == '.' or cand.value == ',':
        point += 30

    if hasattr(c, "prob") and c.value is not None and len(c.value) > 0 and len(cand.value) > 0:
        if c.value == '\'':
            point += 2
        if cand.prob > 0.99 and check_syllable(c.value) and split_syllable_char(c.value)[1] == 'ㅏ':
            point += 2

    return cand.tail + 1 - point

def find_split_pt(img):
    """Find appropriate split point. Namely, the most white column"""
    lidx = 0
    minidx = None
    minidxend = None
    ridx = 32
    fixed = False
    col = np.sum(img, axis=0)

    for i in range(0, 12):
        if col[i] > 2:
            lidx = max(0, i-2)
            break

    for i in range(32, 20):
        if col[i] > 2:
            ridx = min(32, i+2)
            break

    for i in range(12, 20):
        if minidx is None or col[minidx] > col[i]:
            minidx = i
            minidxend = i
        elif col[minidx] == col[i] and not fixed:
            minidxend = i
        else:
            fixed = True
    #print("%d %d, %.4f" % (minidx, minidxend, col[minidx]))
    return (lidx, int((minidx+minidxend)/2), ridx)

def try_split_rotten(char):
    """Try to split an unrecognizable(rotten) character"""
    spt = find_split_pt(char.img)
    limg = reshape_with_margin(char.img[:, spt[0]:spt[1]], 32, 0)
    rimg = reshape_with_margin(char.img[:, spt[1]:spt[2]], 32, 0)
    lpred, rpred = get_pred_batch(np.array([limg, rimg]))

    if lpred.candidate is None and rpred.candidate is None:
        char.value = "?"
        char.rotten = 1
        return
    if lpred.candidate is None:
        lpred.candidate = "?"
        char.rotten = 1
    elif rpred.candidate is None:
        rpred.candidate = "?"
        char.rotten = 1
    char.prob = max(min(lpred.sure, rpred.sure) - 0.4, 0)
    char.value = lpred.candidate + rpred.candidate
    char.tail += 1
    print("%d %d %d %s" % (spt[0], spt[1], spt[2], char.value))
    draw_subplot([limg, rimg], 2, 1)
    return True

def merge_children(clist):
    """Take candidate with lowest 'rotten point'\n
    rotten: number of invalid symbols\n
    tail: refer to eval_tail\n
    prob: lowest probability of symbols
    """
    for c in clist:
        cand = None
        merge_children(c.children)
        for child in c.children:
            if cand is None:
                cand = child
            elif child.rotten_point < cand.rotten_point:
                cand = child

        if cand is None:
            c.tail = 0
            c.rotten = 0
            if c.value is None:
                try_split_rotten(c)
        else:
            c.tail = eval_tail(c, cand)
            # The first character of semantic word does not have prob
            # It only sums up its children
            if hasattr(c, "prob"):
                c.prob = min(c.prob, cand.prob)
                c.rotten = 0
                if c.value is None:
                    try_split_rotten(c)
                c.value += cand.value
                c.rotten += cand.rotten
            else:
                c.prob = cand.prob
                c.value = cand.value
                c.rotten = cand.rotten
        c.rotten_point = c.rotten * 50 + (1-c.prob) * 100 + c.tail

def analyze_sibiling(clist):
    """Analyze relationship between siblings."""
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
    """Analyze given graphs, return semantic analysis result"""
    len_l = 0
    for p in graphs:
        for l in p.lines:
            len_l += 1
    #plt.figure()
    i_l = 0
    for p in graphs:
        for l in p.lines:
            #prev = None
            #plt.subplot(len_l,1, i_l+1)
            #plt.imshow(l.img)
            analyze_recur(l.chars)
            print_recur(i_l, 0, l.chars, True)
            merge_children(l.chars)
            analyze_sibiling(l.chars)
            #for c in l.chars:
            #    print("(%2d, %4d, %4d, %4d) %s" % (i_l, c.pt[0], c.pt[1], c.rotten_point, c.value))
            i_l += 1
    #plt.show()
    return graphs
