from flask import Flask, redirect, url_for, request,jsonify
from svm import SVM_predict
from vgg19 import VGG_predict

app = Flask(__name__)

@app.route("/svm", methods=['POST'])
def SVM_service():
        print("svm")
        music=request.files['file'].filename
        genre=SVM_predict(music)
        return jsonify({"genre":genre})

@app.route("/vgg19", methods=['POST'])
def VGG_service():
        music=request.files['file'].filename
        genre=VGG_predict(music)
        return jsonify({"genre":genre})

if __name__ == '__main__':
    app.run(debug = False,port=5000,host="0.0.0.0")
