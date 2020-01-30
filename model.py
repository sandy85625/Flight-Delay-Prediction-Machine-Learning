
"""Predicting Late Arrivals"""
"""Importing required Libraries"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import pickle

"""Importing Datasets and Locating them"""
dataset = pd.read_csv("/home/sandeep/Downloads/Unisys/uni3.csv")
x = dataset.iloc[:, 4:5]
y = dataset.iloc[:, 5:6]
print(x, y)

"""Feature Scaling"""
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x).reshape(-1, 1)
y = sc_y.fit_transform(y).reshape(-1, 1)

"""Fitting SVR into Dataset"""
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

"""Splitting Datasets into Test and Train Data"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
sc_x_train = StandardScaler()
sc_y_train = StandardScaler()
sc_x_test = StandardScaler()
sc_y_test = StandardScaler()
x_train = sc_x_train.fit_transform(x_train)
y_train = sc_y_train.fit_transform(y_train)
x_test = sc_x_test.fit_transform(x_test)
y_test = sc_y_test.fit_transform(y_test)

"""Plotting Training Dataset"""
plt.scatter(x_train, y_train, color='black')
plt.scatter(x_train, regressor.predict(x_train), color="red")
plt.xlabel("Sch. Departure Time")
plt.ylabel("Actual Departure Time")
plt.show()

"""Making Prediction"""
X = sc_x_test.inverse_transform(x_test)
Y = regressor.predict(X)
y_pred = sc_y_test.inverse_transform(regressor.predict(X))
y = sc_y_test.inverse_transform(y)

"""Downloading model into the Localsystem"""
pickle.dump(regressor, open("model.pkl", 'wb'))

"""Reloading the model"""
pickle.load(open("model.pkl", "rb"))
print(regressor.predict(X))

"""Plotting Predicted Data"""
plt.scatter(X, y_test, color='blue')
plt.scatter(X, y_pred, color="red")
plt.xlabel("Sch. Departure Time")
plt.ylabel("Predicted Departure Time")
plt.show()

"""End"""



