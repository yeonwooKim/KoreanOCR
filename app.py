from flask import Flask, render_template, request
from scipy.ndimage import imread
import preproc
import detect
import chrecog
chrecog.load_ckpt("data/only_valid_160930.ckpt")
import reconst

app = Flask(__name__)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def analysis(img):
    processed = preproc.process(img)
    graphs = detect.get_graphs(processed)
    # TODO: do this in batch, not pred_one
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if c.type != "blank":
                    c.pred = chrecog.get_pred_one(c.img)
    return reconst.build_graphs(graphs)

@app.route('/')
def view_index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def view_upload():
    f = request.files['image']
    blob = f.read()
    size = len(blob)
    f.seek(0)
    img = imread(f)
    analyzed = analysis(img)
    ret = ( "name : %s\n" % f.filename +
            "size : %s\n" % sizeof_fmt(size) +
            "dimension : %d X %d\n" % (img.shape[1], img.shape[0]) +
            "\n<Analysis>\n%s" % analyzed)
    return ret
                           
if __name__ == '__main__':
    app.run()