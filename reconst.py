# reconst.py
# Reconstruction
import chrecog
from detect import CHARTYPE

# 단순히 모든 char를 순서대로 출력
def build_graphs(graphs):
    output = ""
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if c.type == CHARTYPE.CHAR:
                    output += chrecog.get_candidate(c.pred)
                else:
                    output += " "
            output += "\n"
        output += "===="
    return output