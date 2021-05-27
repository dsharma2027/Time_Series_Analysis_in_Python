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
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error 
from math import sqrt 

# read in the dataset, split as train the first 10 entries and rest as test
series = read_csv('seriesdat.csv')
train = series.iloc[0:10]
test = series.iloc[10:]
y_hat = test.copy()

# fit a SES without specifying smooth parameter, so library does it for us
fitx = SimpleExpSmoothing(np.asarray(train['y'])).fit(optimized=True)
print('optimal alpha chosen:', fitx.model.params['smoothing_level'])

# fit a SES with explicitly supplied smooth parameter
fit2 = SimpleExpSmoothing(np.asarray(train['y'])).fit(smoothing_level=0.4, optimized=False)

# print the rmse of training step
rms_train_opt = sqrt(mean_squared_error(train['y'], fitx.fittedvalues)) 
print('rmse for fitted values with optimal alpha: ', rms_train_opt)
rms_train = sqrt(mean_squared_error(train['y'], fit2.fittedvalues)) 
print('rmse for fitted values with supplied alpha: ', rms_train)

# plot training data against fitted values using both models
plt.figure(figsize=(10,3))
plt.plot(fitx.fittedvalues, label = 'fitted opt')
plt.plot(fit2.fittedvalues, label = 'fitted')
plt.plot(train['y'], label='train')
plt.legend(loc='best')
plt.show()

# forecast using both models
y_hat['SES_opt_fcast'] = fitx.forecast(len(y_hat))
y_hat['SES_fcast'] = fit2.forecast(len(y_hat))

# display forecasted values
y_hat

# plot the time series as train, test and forecasted
plt.figure(figsize=(10,3))
plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.plot(y_hat['SES_fcast'], label='SES_fcast')
plt.plot(y_hat['SES_opt_fcast'], label='SES_opt_fcast')
plt.legend(loc='best')
plt.show()

# calculate RMSE of the forecast on test data
rms = sqrt(mean_squared_error(test.y, y_hat.SES_fcast)) 
print('rmse for model with supplied alpha: ', rms)
rms_opt = sqrt(mean_squared_error(test.y, y_hat.SES_opt_fcast)) 
print('rmse for model with optimal alpha: ', rms_opt)

