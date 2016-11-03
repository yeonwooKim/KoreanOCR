from detection.util import CHARTYPE
from chrecog.predict import get_pred_one, get_candidate

def analyze(graphs):
    # TODO: do this in batch, not pred_one
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if c.type != CHARTYPE.BLANK:
                    assert c.type == CHARTYPE.CHAR
                    c.pred = get_pred_one(c.img)
                    c.value = get_candidate(c.pred)
                else:
                    c.value = " "
    return graphs