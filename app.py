# app.py
# 프론트엔드와 각 모듈을 중계
# templates 폴더에 프론트엔드 구현이 되어있음

from flask import Flask, render_template, request
from scipy.ndimage import imread
import preproc
import detection
import chrecog
chrecog.load_ckpt("data/only_valid_160930.ckpt")
import reconst

app = Flask(__name__)

# 파일 사이즈를
# Human readable format으로 변경
# e.g. 102400 -> "100.0KiB"
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# 이미지를 각 모듈에 순서대로 넘겨줌
# 분석된 최종 문자열을 반환
def analysis(img):
    processed = preproc.process(img)
    graphs = detection.get_graphs(processed)
    # TODO: do this in batch, not pred_one
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if c.type != "blank":
                    c.pred = chrecog.get_pred_one(c.img)
    return reconst.build_graphs(graphs)

# 프론트엔드 index
@app.route('/')
def view_index():
    return render_template('index.html')

# Upload 엔트리
# 이미지 파일을 받아 numpy array로 디코드
# 사이즈와 이름 등 기본 정보를 분석 후
# analysis()를 호출
@app.route('/upload', methods=['POST'])
def view_upload():
    f = request.files['image']
    size = len(f.read())
    f.seek(0) # 사이즈 분석을 위해 버퍼를 모두 읽었으므로 초기화
    img = imread(f) # 이미지를 디코드 후 numpy array로 변환
    analyzed = analysis(img)
    ret = ( "name : %s\n" % f.filename +
            "size : %s\n" % sizeof_fmt(size) +
            "dimension : %d X %d\n" % (img.shape[1], img.shape[0]) +
            "\n<Analysis>\n%s" % analyzed)
    return ret

# 웹서버를 통하지 않고
# python 인터프리터로 바로 실행되었을 때
if __name__ == '__main__':
    app.run(debug=True)