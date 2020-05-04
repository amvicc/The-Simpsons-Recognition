import os
from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__,  static_folder="build\\static", template_folder="build")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/upload', methods=['POST'])
def file_upload():
    if os.path.isfile('files\\img.jpg'):
        os.remove("files\\img.jpg")
    if os.path.isfile('result.txt'):
        os.remove('result.txt')

    target = os.path.join('files')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    destination = "/".join([target, 'img.jpg'])
    file.save(destination)
    os.system('main.py')
    if os.path.isfile('result.txt'):
        f = open("result.txt", "r")
        response = f.read()
        f.close()
        return response
    else:
        response = 'No Simpson on photo!'
        return response


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
