# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:46:05 2019

@author: ssridhar
"""
# set current working directory
import os
print("This is the cwd: ", os.getcwd())
os.chdir('D:\\ssridhar\\research\\AnalyticsBigdataML\\ML Course\\Praxis\\July 2020\\TSF\\Session 2 - Exponential Smoothing Methods')
print("This is the cwd: ", os.getcwd())

# import libraries
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error 
from math import sqrt 

# import dataset, split into train and test
series = read_csv('seriesdat.csv')
train = series.iloc[0:10]
test = series.iloc[10:]


# using Holt's linear model predict for test set
y_hat = test.copy()
fit = Holt(np.asanyarray(train['y'])).fit(smoothing_level=0.4, smoothing_slope=0.7, optimized=False)
y_hat['Holt_linear'] = fit.forecast(len(test))
y_hat

# plot the forecast
plt.figure(figsize=(10,3))
plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.plot(y_hat['Holt_linear'], label='Holt linear')
plt.legend(loc='best')
plt.show()

# compute the error
rmse=sqrt(mean_squared_error(test.y, y_hat['Holt_linear']))
print(rmse)

# same as above, now using the ExponentialSmoothing library
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(np.asarray(train['y']), trend='add', seasonal=None)
fit1 = model.fit(smoothing_level=0.4, smoothing_slope=0.7, optimized=False)
y_hat['DES'] = fit1.forecast(len(test))
y_hat

plt.figure(figsize=(10,3))
plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.plot(y_hat['Holt_linear'], label='Holt linear')
plt.plot(y_hat['DES'], label='DES')
plt.legend(loc='best')
plt.show()

rmse=sqrt(mean_squared_error(test.y, y_hat['DES']))
print(rmse)


# damped trend (optional)
fit = Holt(np.asanyarray(train['y']), damped=True).fit(smoothing_level=0.4, smoothing_slope=0.7, damping_slope=0.7)
y_hat['Holt_linear_damped'] = fit.forecast(len(test))
y_hat
plt.figure(figsize=(10,3))
plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.plot(y_hat['Holt_linear'], label='Holt linear')
plt.plot(y_hat['Holt_linear_damped'], label='Holt linear damped')
plt.legend(loc='best')
plt.show()

rmse=sqrt(mean_squared_error(test.y, y_hat['Holt_linear_damped']))
print(rmse)

