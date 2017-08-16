#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:35:44 2017

@author: vinay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

# Independent Variables and Dependent Variable

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Categorical Data
# Encoding Independent Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variables Trap
X = X[:, 1:]

# Train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Predict using Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict test results
Y_pred = regressor.predict(X_test)

# Building optimal model using backward ellimination
from statsmodel.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)