from detection.util import CHARTYPE
from chrecog.predict import get_pred_one, get_candidate

def merge_char_if_can(prev, c):
    if prev is None:
        return
    
    if prev.type == CHARTYPE.BLANK or c.type == CHARTYPE.BLANK:
        return

    if prev.value == "'" and c.value == "'":
        prev.value = ""
        c.value = '"'

def analyze(graphs):
    # TODO: do this in batch, not pred_one
    for p in graphs:
        for l in p.lines:
            prev = None
            for i in range(len(l.chars)):
                c = l.chars[i]
                if c.type != CHARTYPE.BLANK:
                    assert c.type == CHARTYPE.CHAR
                    c.pred = get_pred_one(c.img / 255)
                    c.value = get_candidate(c.pred)
                    merge_char_if_can(prev, c)
                else:
                    c.value = " "
                prev = c
    return graphs