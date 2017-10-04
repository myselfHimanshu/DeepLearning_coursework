#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:33:15 2017

@author: Himanshu
@title: Titanic: Machine Learning from Disaster
"""

#import libraries
import pandas as pd
import numpy as np

#First get the training data
train_df = pd.read_csv("./train.csv")
X = train_df[:,]

