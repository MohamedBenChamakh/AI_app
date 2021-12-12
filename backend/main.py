from flask import Flask, redirect, url_for, request,jsonify
from svm import SVM_predict
from vgg19 import VGG_predict

app = Flask(__name__)

@app.route("/svm", methods=['POST'])
def SVM_service():
    if request.method == 'POST':
        json_data = request.get_json(force=True) 
        wav_music=json_data['wav_music']
        genre=SVM_predict(wav_music)
        return jsonify({"genre":genre})


@app.route("/vgg19", methods=['POST'])
def VGG_service():
    if request.method == 'POST':
        json_data = request.get_json(force=True) 
        wav_music=json_data['wav_music']
        genre=VGG_predict(wav_music)
        return jsonify({"genre":genre})

if __name__ == '__main__':
    app.run(debug = True,port=5000,host="0.0.0.0")
