from flask import Flask

app = Flask(__name__)



@app.route("/", methods=['POST'])
def SVM_service(wav_music):
  
    return "<h1 style='color:blue'>Hello There!</h1>"


if __name__ == '__main__':
    app.run()
