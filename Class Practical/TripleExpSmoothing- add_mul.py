# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:46:05 2019

@author: ssridhar
"""

# import libraries
import numpy as np
from pandas import read_csv
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
from sklearn.metrics import mean_squared_error 
from math import sqrt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
os.chdir('D:\\ssridhar\\research\\AnalyticsBigdataML\\ML Course\\Praxis\\July 2020\\TSF\\Session 2 - Exponential Smoothing Methods')
print("This is the cwd: ", os.getcwd())


# read in the ts data
series = read_csv('seriesseasonal.csv')

# plot the full series
plt.figure(figsize=(10,3))
plt.plot(series['y'], label='series')
plt.show()


# Using ExponentialSmoothing
# this is based on Real Statistics worksheet "Holt-Winters 4a"

# we are building a model [L + T]* S template and passing 4 as length of season cycle
model = ExponentialSmoothing(np.asarray(series['y']), seasonal_periods=4, trend='add', seasonal='mul')

# fit a model giving these inputs. the strange numbers are a result of optimization.
fit1 = model.fit(smoothing_level=0.2, smoothing_slope=0.2, smoothing_seasonal=0.2, optimized=False)


# how to get info about the model, fitted values
fit1.sse # SSE on training data
fit1.level # level values fitted
fit1.slope # trend values fitted
fit1.season # season value fitted
fit1.fittedvalues # fitted values
fit1.fcastvalues # one time step ahead forecast
fit1.fittedfcast # fitted and one time step ahead forecast
fit1.summary() # model parameters, e.g. alpha, beta etc used or optim values

# let us reconcile the SSE output.
# this calculates RMSE comparing original data against fitted values
rmse=sqrt(mean_squared_error(series['y'], fit1.fittedvalues))
print(rmse)
# to compare with SSE, first we need to divide by 16 and take sqrt
np.sqrt(1/16 * fit1.sse)

# lets see how the original data and fitted values are visually
plt.figure(figsize=(10,3))
plt.plot(series['y'], label='series')
plt.plot(fit1.fittedvalues, label='fitted_values')
plt.legend(loc='best')
plt.show()

# let us try splitting as train and test and try to forecast the test time periods

# split data training and test, create a dataframe with copy of test
train = series.iloc[0:5]
test = series.iloc[5:]
y_hat = test.copy() # copy to a dataframe so we can add columns later on

# fit model on just training data
model = ExponentialSmoothing(np.asarray(train['y']), seasonal_periods=4, trend='add', seasonal='mul')
fit1 = model.fit(smoothing_level=0.2, smoothing_slope=0.2, smoothing_seasonal=0.2, optimized=False)

# forecast for as many time periods as in test data
y_hat['TES_add_mul'] = fit1.forecast(len(test))

# see what is in this dataframe
y_hat # compare these values against the forecast in real-statistics

# plot the splits as train, test and forecasted values
plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.plot(y_hat['TES_add_mul'], label='forecast')
plt.legend(loc='best')
plt.show()

# how well did the model perform against test?
rmse=sqrt(mean_squared_error(test.y, y_hat['TES_add_mul']))
print(rmse)


# optimized version
# this combination is probably the best for reducing MSE
fitx = model.fit(smoothing_level=0.0426, smoothing_slope=0.355, smoothing_seasonal=0.161, optimized=False, initial_level=14.25, initial_slope=1.875, use_brute=False)
y_hat['TES_fitx'] = fitx.forecast(len(test))
rmse=sqrt(mean_squared_error(test.y, y_hat['TES_fitx']))
print(rmse)
fitx.fittedvalues
fitx.forecast(len(test))
fitx.summary()


plt.plot(train['y'], label='train')
plt.plot(test['y'], label='test')
plt.plot(y_hat['TES_fitx'], label='forecast')
plt.legend(loc='best')
plt.show()




# reconciling [L + T] * S
import pandas as pd
df = pd.DataFrame({'level':fit1.level, 'slope': fit1.slope, 'season':fit1.season,'fittedvalues':fit1.fittedvalues})
df
# reconcile this against how real-statistics has calculated forecast

# there are minor logic differences
# if curious inspect the code of this library at
# https://www.statsmodels.org/dev/_modules/statsmodels/tsa/holtwinters.html#ExponentialSmoothing.fit



import statsmodels.api as sm
sm.tsa.seasonal_decompose(train.y, model='add', freq=2).plot()
#result = sm.tsa.stattools.adfuller(Train.count)
plt.show()
