import os
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/upload', methods=['POST'])
def file_upload():
    target = os.path.join('files')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    destination = "/".join([target, 'i.venvmg.jpg'])
    file.save(destination)
    # session['uploadFilePath'] = destination
    os.system('main.py')
    os.remove("files\\img.jpg")
    f = open("result.txt", "r")
    response = f.read()
    f.close()
    os.remove('result.txt')
    return response


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
