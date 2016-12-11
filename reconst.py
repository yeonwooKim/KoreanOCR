# reconst.py
# Reconstruction

import json

def build_json(imgname, graphs):
    out = dict()
    out["id"] = imgname
    labels = []
    for graph in graphs:
        for line in graph.lines:
            label = dict()
            label["x"] = line.rect[0]
            label["y"] = line.rect[1]
            label["width"] = line.rect[2] - line.rect[0]
            label["height"] = line.rect[3] - line.rect[1]

            linetxt = ""
            for c in line.chars:
                linetxt += c.value
            label["desc"] = linetxt
            labels.append(label)

    out["labels"] = labels
    return json.dumps(out, ensure_ascii=False)


def flush_line_buf(line_buf):
    line_buf.sort(key=lambda line: line.rect[0])
    ret = "".join([line.txt for line in line_buf])
    line_buf.clear()
    return ret

def valid_line(linetext):
    for c in linetext:
        if c != ' ': return True
    return False

# 단순히 모든 char를 순서대로 출력
def build_graphs(graphs):
    pages = assemble_page(graphs)
    text = []
    for lines in pages:
        text.extend(assemble_line(lines))
    output = ""
    for linetext in text:
        if valid_line(linetext):
            output += linetext + "\n"
    return output

def assemble_page(graphs):
    lines = []
    for graph in graphs:
        lines.extend(graph.lines)

    docw = max([line.rect[2] for line in lines]) - min([line.rect[0] for line in lines])
    doch = max([line.rect[3] for line in lines]) - min([line.rect[1] for line in lines])
    if docw < doch: return [lines]
    print("is it a page..?")
    xavg = sum([line.rect[0] for line in lines]) / len(lines)
    page0 = []
    page1 = []
    for line in lines:
        if line.rect[0] < xavg: page0.append(line)
        else: page1.append(line)
    return [page0, page1]

def assemble_line(lines):
    text = []
    for l in lines:
        linetxt = ""
        for c in l.chars:
            linetxt += c.value
        l.txt = linetxt
        #print("[%4d %4d %4d %4d] %s" % (l.rect + (linetxt,)))
    
    lines.sort(key=lambda line: line.rect[1])

    prev_y = None
    line_buf = []
    for l in lines:
        if prev_y is not None and prev_y+10 < l.rect[1]:
            text.append(flush_line_buf(line_buf))
        line_buf.append(l)
        prev_y = l.rect[1]
    text.append(flush_line_buf(line_buf))
    return text
