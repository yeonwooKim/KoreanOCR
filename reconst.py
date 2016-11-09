# reconst.py
# Reconstruction

# 단순히 모든 char를 순서대로 출력
def build_graphs(graphs):
    output = ""
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                output += c.value
            output += "\n"
        #output += "====\n"
    return output