from detection.util import CHARTYPE
import chrecog

def analyze(graphs):
    # TODO: do this in batch, not pred_one
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if c.type != CHARTYPE.BLANK:
                    assert c.type == CHARTYPE.CHAR
                    c.pred = chrecog.get_pred_one(c.img)
    return graphs