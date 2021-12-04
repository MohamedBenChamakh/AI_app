import os
import pandas as pd # Pour le dataframe
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

import pickle
from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

def get_features(img_path):
    img = image.load_img(img_path, target_size=(432, 288)).resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    return list(flatten[0])

X = []

bike_plots = []
car_plots = []

genre=['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
filenames=[]

df = pd.read_csv("../data/features_30_sec.csv")
df = df.drop(labels='filename', axis=1)
labels=df.iloc[:,-1]

encoder=LabelEncoder()
labels=encoder.fit_transform(labels)

for g in genre:
    for (_,_,filenames) in os.walk('../data/images_original/'+g):
        for f in filenames:
         X.append(get_features('../data/images_original/'+g +'/'+ f))

print(len(X))
print(len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, labels[:-1], test_size=0.30, random_state=42, stratify=labels[:-1])

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train)
# modelname = 'model_linearSvc.sav'
# pickle.dump(model, open(clf, 'wb'))
predicted = clf.predict(X_test)

# get the accuracy
print (accuracy_score(y_test, predicted))
