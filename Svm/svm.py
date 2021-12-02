import pandas as pd # Pour le dataframe
import numpy as np # Pour la normalisation et calculs de moyenne
import matplotlib.pyplot as plt # Pour la visualisation

import librosa # Pour l'extraction des features et la lecture des fichiers wav
import librosa.display # Pour récupérer les spectrogrammes des audio
import librosa.feature

import time

import os # C'est ce qui va nous permettre d'itérer sur les fichiers de l'environnement de travail
import sklearn
import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, validation_curve, RandomizedSearchCV # Split de dataset et optimisation des hyperparamètres

from sklearn.svm import SVC # SVM
import pickle
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import csv

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def svm():
    print("RBF Kernel")
    df = pd.read_csv("../Data/features_30_sec.csv")
    df = df.drop(labels='filename', axis=1)
    #print(df.head())
    #print(df.shape)

    labels=df.iloc[:,-1]
    #print(labels)
    encoder=LabelEncoder()
    labels=encoder.fit_transform(labels)
    #print(labels)
   
    standardizer=StandardScaler()
    data=standardizer.fit_transform(np.array(df.iloc[:,:-1],dtype=float))
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.3)
    print('length of data_train:',len(data_train))
    print('length of data_test:',len(data_test))
    model = SVC(kernel='poly', C=50,degree=3)
    model.fit(data_train, labels_train)
    modelname = 'model_svc.sav'
    #os.remove(modelname)
    #time.sleep(0.5)
    pickle.dump(model, open(modelname, 'wb'))
    print('Train score : ', model.score(data_train,labels_train))
    print('Test score : ', model.score(data_test,labels_test))
    pred = model.predict(data_test)
    #print(pred)
    #print(confusion_matrix(labels_test, pred))
    #print(classification_report(labels_test, pred))
    #grid_search(data_train,labels_train)

def grid_search(data_train,labels_train):
    tuned_parameters = [{'kernel': ['rbf'],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'],'C':[1, 10, 100, 1000],'degree':[1,2,3,4]}]
    grid_svm = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy', n_jobs=-1, cv=None,verbose=2)
    grid_svm.fit(data_train, labels_train)
    print(grid_svm.best_params_)

def predict(audio):
    #audio="reggae.00000.wav"
    csv_file = csv.reader(open("../Data/features_30_sec.csv", "r"), delimiter=",")
    next(csv_file, None)

    data=[]
    for row in csv_file:
        #if current rows 2nd value is equal to input, print that row
        if audio == row[0]:
            data=np.array([row[1:-1]])
    if len(data) >0 :       
        svm = joblib.load('model_svc.sav')
        print("----------------------------------- Predicted Labels -----------------------------------\n")
        preds = svm.predict(data)
        #print(preds)
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
        #print(preds)
        func = switcher.get(preds[0], lambda: "Invalid gender")
        print("Audio : ",audio)
        print("Genre: ",func)
        print("")
        print("----------------------------------------------------------------------------------------")


svm()
#predict("metal.00000.wav")


csv_file = csv.reader(open("../Data/features_30_sec.csv", "r"), delimiter=",")
next(csv_file, None)

for row in csv_file:
    predict(row[0])
    time.sleep(0.5)
