from flask import Flask, render_template, request
app = Flask(__name__)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

@app.route('/')
def view_index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def view_upload():
    f = request.files['image']
    ret = "name : %s\n" % f.filename
    blob = f.read()
    size = len(blob)
    ret += "size : %s" % sizeof_fmt(size)
    return ret
                           
if __name__ == '__main__':
    app.run()