#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:13:34 2017

@author: vinay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# np.set_printoptions(threshold=np.inf)
# create dependent and independent variables matrix

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Split the dataset for training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Machine has learned. Now we can predict

# Predict
Y_pred = regressor.predict(X_test)

# Plot
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Years of Experience vs Salary (Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Years of Experience vs Salary (Test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()