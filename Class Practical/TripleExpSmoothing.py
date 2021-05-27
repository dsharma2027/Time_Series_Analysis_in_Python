# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:46:05 2019

@author: ssridhar
"""
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
import os
os.chdir('D:\\ssridhar\\research\\AnalyticsBigdataML\\ML Course\\Praxis\\July 2020\\TSF\\Session 2 - Exponential Smoothing Methods')
print("This is the cwd: ", os.getcwd())

series = read_csv('seriesseasonal.csv')
train = series.iloc[0:10]
test = series.iloc[10:]
y_hat = test.copy()

# plot the full series
plt.figure(figsize=(10,3))
plt.plot(series['y'], label='series')
plt.show()

# plot the splits as train and test
plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.legend(loc='best')
plt.show()


# Using ExponentialSmoothing
# this is based on Real Statistics worksheet "Holt-Winters 4a"
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(np.asarray(train['y']), seasonal_periods=4, trend='add', seasonal='mul')
fit1 = model.fit(smoothing_level=0.0426, smoothing_slope=0.355, smoothing_seasonal=0.161, optimized=False)
y_hat['TES_add_mul'] = fit1.forecast(len(test))
y_hat

plt.figure(figsize=(10,3))
plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.plot(y_hat['TES_add_mul'], label='TES_add_mul')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error 
from math import sqrt 
rmse=sqrt(mean_squared_error(test.y, y_hat['TES_add_mul']))
print(rmse)

import statsmodels.api as sm
sm.tsa.seasonal_decompose(train.y, model='add', period=4).plot()
#result = sm.tsa.stattools.adfuller(Train.count)
plt.show()

# additive model
# corresponds to real-statistics worksheet "Holt Winters 5"
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(np.asarray(train['y']), seasonal_periods=4, trend='add', seasonal='add')
fit1 = model.fit(smoothing_level=0.015532, smoothing_slope=1, smoothing_seasonal=0.23088, optimized=False)
y_hat['TES_add'] = fit1.forecast(len(test))
y_hat
rmse=sqrt(mean_squared_error(test.y, y_hat['TES_add']))
print(rmse)

# this part goes deeper into use of the library, optional

model.initial_values()
fit1 = model.fit(smoothing_level=0.015532, smoothing_slope=1, smoothing_seasonal=0.23088, optimized=False, initial_level=14.25, initial_slope=1.875)
