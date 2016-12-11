# reconst.py
# Reconstruction

def flush_line_buf(line_buf):
    line_buf.sort(key=lambda line: line.rect[0])
    ret = "".join([line.txt for line in line_buf]) + "\n"
    line_buf.clear()
    return ret

# 단순히 모든 char를 순서대로 출력
def build_graphs(graphs):
    lines = assemble_line(graphs)
    lines.sort(key=lambda line: line.rect[1])
    output = ""
    prev_y = None
    line_buf = []
    for l in lines:
        if prev_y is not None and prev_y+10 < l.rect[1]:
            output += flush_line_buf(line_buf)
        line_buf.append(l)
        prev_y = l.rect[1]
    output += flush_line_buf(line_buf)
    return output

def assemble_line(graphs):
    lines = []
    for p in graphs:
        for l in p.lines:
            linetxt = ""
            for c in l.chars:
                linetxt += c.value
            l.txt = linetxt
            lines.append(l)
            print("[%4d %4d %4d %4d] %s" % (l.rect + (linetxt,)))
    return lines