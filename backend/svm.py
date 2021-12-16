import pandas as pd # Pour le dataframe
import numpy as np # Pour la normalisation et calculs de moyenne
import matplotlib.pyplot as plt # Pour la visualisation

import librosa # Pour l'extraction des features et la lecture des fichiers wav
import librosa.display # Pour récupérer les spectrogrammes des audio
import librosa.feature


import os # C'est ce qui va nous permettre d'itérer sur les fichiers de l'environnement de travail

from time import sleep
import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.svm import SVC 
import pickle
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler

import csv

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def svm():
    print("RBF Kernel")
    df = pd.read_csv("data/features_3_sec.csv")
    df = df.drop(labels='filename', axis=1)

    labels=df.iloc[:,-1]
 
    encoder=LabelEncoder()
    labels=encoder.fit_transform(labels)

   
    standardizer=StandardScaler()
    data=standardizer.fit_transform(np.array(df.iloc[:,:-1],dtype=float))

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.33)
    print('length of data_train:',len(data_train))
    print('length of data_test:',len(data_test))
    model = SVC(kernel='rbf', C=100)
    model.fit(data_train, labels_train)
    modelname = 'models/model_svm.sav'
    pickle.dump(model, open(modelname, 'wb'))
    print("Accuracy on training set: {:.3f}".format(model.score(data_train, labels_train)))
    print("Accuracy on test set: {:.3f}".format(model.score(data_test, labels_test)))
    print('Train score : ', model.score(data_train,labels_train))
    print('Test score : ', model.score(data_test,labels_test))
    pred = model.predict(data_test)
    #print("Accuracy:",metrics.accuracy_score(labels_test, pred))
    #print(pred)
    #print(confusion_matrix(labels_test, pred))
    #print(classification_report(labels_test, pred))
    #grid_search(data_train,labels_train)
    return {"Train_score": model.score(data_train,labels_train),"Test_score":model.score(data_test,labels_test),"Accuracy":metrics.accuracy_score(labels_test, pred)}

def grid_search(data_train,labels_train):
    tuned_parameters = [{'kernel': ['rbf'],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'],'C':[1, 10, 100, 1000],'degree':[1,2,3,4]}]
    grid_svm = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy', n_jobs=-1, cv=None,verbose=2)
    grid_svm.fit(data_train, labels_train)
    print(grid_svm.best_params_)

def SVM_predict(audio):
    df = pd.read_csv("data/features_30_sec.csv")
    df = df.drop(labels='filename', axis=1)
    standardizer=StandardScaler()
    data=standardizer.fit_transform(np.array(df.iloc[:,:-1],dtype=float))
    csv_file = csv.reader(open("data/features_30_sec.csv", "r"), delimiter=",")
    i=0
    for row in csv_file:
        if audio == row[0]:
            data=np.array([data[i]],dtype=float)
        i+=1
    if len(data) >0 :       
        svm = joblib.load('models/model_svm.sav')
        print("----------------------------------- Predicted Labels -----------------------------------\n")
        preds = svm.predict(data)
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
        func = switcher.get(preds[0], lambda: "Invalid gender")
        print("Audio : ",audio)
        print("Genre: ",func)
        print("")
        print("----------------------------------------------------------------------------------------")
        return func
