import chrecog

def build_graphs(graphs):
    output = ""
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if c.type != "blank":
                    output += chrecog.get_candidate(c.pred)
                else:
                    output += " "
            output += "\n"
        output += "===="
    return output