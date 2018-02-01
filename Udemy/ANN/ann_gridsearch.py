#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:03:35 2017

@title: Artificial Neural Network
@author: Himanshu Panwar
"""
#ANN
#Install theano
#Install Tensorflow
#Install Keras

#print("Hello world!!!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Part 1 : Data Preprocessing
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding the categorical features(the independent variables)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Part 2: Creating the ANN
#Creating the classifier

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


#Initaializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(activation="relu", kernel_initializer="uniform", input_dim=11, units=6))
classifier.add(Dropout(rate = 0.1))

#Adding the second hidden layer
classifier.add(Dense(activation="relu", kernel_initializer="uniform", units=6))
classifier.add(Dropout(rate = 0.1))

#Adding the output layer
classifier.add(Dense(activation="sigmoid", kernel_initializer="uniform", units=1))

#Compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

#Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#Part 3: Making the predictions and evaluating the model

#Prediction of the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Homework
#Predict for a customer
"""
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

#new_prediction = classifier.predict(sc.transform(np.array([[0., 0., 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
#new_prediction = (new_prediction>0.5)


#Part 4: Evaluting, Improving and tunning the ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation="relu", kernel_initializer="uniform", input_dim=11, units=6))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation="relu", kernel_initializer="uniform", units=6))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation="sigmoid", kernel_initializer="uniform", units=1))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()


#Imporoving the ANN
#Dropout regularization to reduce the overfitting the dataset



#Tunning the ANN
#Get the best hyperparameters, using gridsearch



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation="relu", kernel_initializer="uniform", input_dim=11, units=6))
    #classifier.add(Dropout(0.1))
    classifier.add(Dense(activation="relu", kernel_initializer="uniform", units=6))
    #classifier.add(Dropout(0.1))
    classifier.add(Dense(activation="sigmoid", kernel_initializer="uniform", units=1))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
params = {"batch_size":[25, 32],
           "epochs":[100, 500],
           "optimizer":["adam","rmsprop"]}
 
 
grid_search = GridSearchCV(estimator = classifier,
                            param_grid = params,
                            scoring = "accuracy",
                            cv = 10)
 
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_


























