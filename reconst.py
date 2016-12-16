# reconst.py
# Reconstruction

import json

def build_json(imgname, graphs):
    '''Build json object result'''
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
    '''Flush gathered lines with sorted to x position'''
    line_buf.sort(key=lambda line: line.rect[0])
    ret = "".join([line.txt for line in line_buf])
    #print([line.rect[1] for line in line_buf])
    line_buf.clear()
    return ret

def valid_line(linetext):
    '''Does this line contain meaningful value?'''
    for c in linetext:
        if c != ' ': return True
    return False

# 단순히 모든 char를 순서대로 출력
def build_graphs(graphs):
    '''Build grpahs according to lines, words, etc.'''
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
    '''Check if the paragraph in multi-paged structure'''
    lines = []
    for graph in graphs:
        lines.extend(graph.lines)

    if len(lines) == 0:
        return [lines]

    docw = max([line.rect[2] for line in lines]) - min([line.rect[0] for line in lines])
    doch = max([line.rect[3] for line in lines]) - min([line.rect[1] for line in lines])
    rightpage_count = 0
    for line in lines:
        if line.rect[0] > docw / 2:
            rightpage_count += 1
    if docw < doch or rightpage_count < len(lines) / 3:
        return [lines]
    print("is it a page..?")
    page0 = []
    page1 = []
    for line in lines:
        if line.rect[0] < docw / 2: page0.append(line)
        else: page1.append(line)
    return [page0, page1]

def assemble_line(lines):
    '''Assemble characters into a complete line'''
    text = []
    heights = []
    for line in lines:
        linetxt = ""
        for c in line.chars:
            linetxt += c.value
        line.txt = linetxt
        heights.append(line.rect[3] - line.rect[1])
        #print("[%4d %4d %4d %4d] %s" % (l.rect + (linetxt,)))
    
    lines.sort(key=lambda line: line.rect[1])
    avg_height = sum(heights) / len(heights)
    #print(avg_height)

    prev_y = None
    line_buf = []
    for l in lines:
        if prev_y is not None and prev_y+avg_height < l.rect[1] + l.rect[3]:
            text.append(flush_line_buf(line_buf))
        line_buf.append(l)
        prev_y = l.rect[1] + l.rect[3]
    text.append(flush_line_buf(line_buf))
    return text
