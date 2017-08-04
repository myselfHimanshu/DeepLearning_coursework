#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:50:01 2017
@author: Himanshu
"""

#Recurrent Neural Network

#Part 1 :Data Preprocessing

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
training_set = pd.read_csv("./Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#Getting the inputs and the outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

#Reshaping
X_train = np.reshape(X_train, (1257,1,1))

#Part 2 : Building the RNN

#Importing the libraries and the modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initializing the RNN
regressor = Sequential()

#Adding the input layer and LSTM 
regressor.add(LSTM(units = 4, activation = "sigmoid", input_shape = (None, 1)))

#Adding the outp layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = "adam", loss = "mean_squared_error")

#Fitting the regressor to the Training Set
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

#Part 3 : Making the predictions and visualizing the results

#Getting the real sock price  of 2017
test_set = pd.read_csv("./Google_Stock_Price_Test.csv")
real_stock_price = test_set.iloc[:,1:2].values

#Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.fit_transform(inputs)

inputs = np.reshape(inputs, (20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visulazing the results
plt.plot(real_stock_price, color = "red", label = "Real Google Stock Price")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()



#HOMEWORK

#Gettint the real google stock price of 2012-16
real_stock_price_train = pd.read_csv("Google_Stock_Price_Train.csv")
real_stock_price_train = real_stock_price_train.iloc[:,1:2].values

#Getting the predicted price of 2012-16
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train =sc.inverse_transform(predicted_stock_price_train)

#Visualizing the results
plt.plot(real_stock_price_train, color = "red", label = "Real Google Stock Price")
plt.plot(predicted_stock_price_train, color = "blue", label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()

 
#Part 4 : Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))




#Instead of 1 timestamp add more
"""
# Creating a data structure with 20 timesteps and t+1 output
X_train = []
y_train = []
for i in range(20, 1258):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
"""






