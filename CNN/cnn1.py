#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:20:09 2017

@author: Himanshu
@title: CNN
"""

#Convolutional Neural Network
#Install Theano
#Install Thensorflow
#Install Keras

#Part 1: Building the CNN

#Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D 
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1 : Convolution 
classifier.add(Conv2D(32, (3, 3), input_shape = (64,64,3), activation = 'relu'))
  
#Step 2 : Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))


#Step 3: Flattening
classifier.add(Flatten())

#Step 4: Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Part 2: Fitting the CNN to the image

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

test_datagen = ImageDataGenerator(
            rescale=1./255
        )

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode = 'binary'
        )

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode = 'binary'
        ) 

classifier.fit_generator(
            training_set,
            steps_per_epoch=8000,
            epochs = 25,
            validation_data=test_set,
            validation_steps=2000
        )















 










 