from flask import Flask, redirect, url_for, request,jsonify
from svm import *


app = Flask(__name__)

@app.route("/svm", methods=['POST'])
def SVM_service():
    if request.method == 'POST':
        json_data = request.get_json(force=True) 
        wav_music=json_data['wav_music']
        
        genre=predict(wav_music)
        return jsonify({"genre":genre})

if __name__ == '__main__':
    app.run(debug = True)
