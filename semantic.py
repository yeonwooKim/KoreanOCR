import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'Greys'
from detection.util import CHARTYPE
import numpy as np

def print_recur(i_l, indent, clist, force_print=False):
    for c in clist:
        if c.type != CHARTYPE.BLANK:
            if force_print or indent > 0 or len(c.children) > 0:
                print("%s (%2d, %4d, %4d)" % (' ' * indent, i_l, c.pt[0], c.pt[1]), end=" ")
                print("%.3f %s" % (c.prob, c.value))
        else:
            if force_print or indent > 0 or len(c.children) > 0:
                print("%s (%2d, %4d, %4d)" % (' ' * indent, i_l, -1, -1), end=" ")
                print("%.3f %s" % (c.prob, c.value))
        if len(c.children) > 0:
            print_recur(i_l, indent + 4, c.children, True)

def analyze_recur(clist):
    prev = None
    for c in clist:
        if c.type != CHARTYPE.BLANK:
            c.prob, c.value = c.pred.sure, c.pred.candidate
        else:
            c.prob, c.value = 1, " "
        if len(c.children) > 0:
            analyze_recur(c.children)
        prev = c
        
def kill_bad_children(clist):
    if len(clist) == 0:
        return
    for c in clist:
        kill_bad_children(c.children)
        # If one of them is bad, abandon all
        for child in c.children:
            if child.value == "" and len(child.children) == 0 :
                c.children = []
                break;

def consume_children(c):
    c.prob = min([child.prob for child in c.children])
    c.value = ''.join([child.value for child in c.children])
    c.children = []

def kill_bad_parent(clist):
    if len(clist) == 0:
        return
    for c in clist:
        # Has no children, nothing to do
        if len(c.children) == 0:
            continue
        kill_bad_parent(c.children)
        if c.value == '' or (c.prob < min([child.prob for child in c.children]) - 0.1):
            consume_children(c)

# 마침표나 쉽표로 나눌 만한 건 나누자
def split_periods(clist):
    for c in clist:
        if len(c.children) == 2:
            if c.children[1].value == '.' or c.children[1].value == ',':
                 consume_children(c)

def merge_with_sibiling(clist):
    prev = None
    for c in clist:
        if prev is None:
            prev = c
            continue
        if prev.value == '\'' and c.value == '\'':
            prev.value = ''
            c.value = '"'
        prev = c

def analyze_pedigree(clist):
    kill_bad_children(clist)
    kill_bad_parent(clist)

def analyze_linear(clist):
    merge_with_sibiling(clist)
    split_periods(clist)

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
            analyze_pedigree(l.chars)
            analyze_linear(l.chars)
            print_recur(i_l, 0, l.chars, True)
            i_l+=1
    #plt.show()
    return graphs