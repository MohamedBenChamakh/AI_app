import csv
import os
import pandas as pd 
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

import joblib
import pickle
from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score



def get_features(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    return list(flatten[0])

def vgg19():
    X = []
    genre=['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

    filenames=[]

    df = pd.read_csv("../data/features_30_sec.csv")
    df = df.drop(labels='filename', axis=1)
    labels=df.iloc[:,-1]
    encoder=LabelEncoder()
    labels=encoder.fit_transform(labels)
    
    for g in genre:
        for (_,_,filenames) in os.walk('../data/images_original/'+g):
            for f in filenames:
                X.append(get_features('../data/images_original/'+g +'/'+ f,model))

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.08, random_state=42, stratify=labels)

    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X_train, y_train)

    modelname = 'models/model_vgg19.sav'
    pickle.dump(clf, open(modelname, 'wb'))
    print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
    print('Train score : ', clf.score(X_train,y_train))
    print('Test score : ', clf.score(X_test,y_test))

def VGG_predict(wav_music):
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)
    csv_file = csv.reader(open("../data/features_30_sec.csv", "r"), delimiter=",")
    data=[]
    for row in csv_file:
        if wav_music == row[0]:
            data=get_features('../data/images_original/'+row[-1] +'/'+ wav_music.replace(".","",1).replace("wav","png",1),model)
    
    if len(data) >0 :       
            svm = joblib.load('models/model_vgg19.sav')
            print("----------------------------------- Predicted Labels -----------------------------------\n")
            predicted = svm.predict([data])
            switcher = {
                0:"blues",
                1: "classical",
                2: "country",
                3: "disco",
                4: "hiphop",
                5: "jazz",
                6: "metal",
                7: "pop",
                8: "reggae",
                9: "rock",
            }
            print(predicted)
            func = switcher.get(predicted[0], lambda: "Invalid gender")
            print("Audio : ",wav_music)
            print("Genre: ",func)
            print("")
            print("----------------------------------------------------------------------------------------")
            return func
    # get the accuracy
    #print (accuracy_score(y_test, predicted))

#predict("reggae.00000.wav")
#vgg19()