# reconst.py
# Reconstruction
from chrecog import get_candidate
from detection.util import CHARTYPE

# 단순히 모든 char를 순서대로 출력
def build_graphs(graphs):
    output = ""
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if c.type == CHARTYPE.CHAR:
                    output += get_candidate(c.pred)
                else:
                    output += " "
            output += "\n"
        output += "====\n"
    return output