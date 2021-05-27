# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:03:01 2019

@author: ssridhar
"""
import os
os.chdir('D:\\ssridhar\\research\\AnalyticsBigdataML\\ML Course\\Praxis\\July 2020\\TSF\\Visualizing Time Series data')

import matplotlib.pyplot as plt
import pandas as pd
stock = pd.read_csv('GOOG.csv', header=None, delimiter=',')
stock.columns = ['date','price']
stock['date'] = pd.to_datetime(stock['date'], format='%d-%m-%Y')
indexed_stock = stock.set_index('date')
ts = indexed_stock['price']
plt.plot(ts)
plt.show()

plt.plot(ts, 'bo--', linewidth=2, markersize=12)
plt.show()

# one more way, this is better than the previous one
plt.plot(ts, color='green', marker='o', linestyle='dashed',      linewidth=2, markersize=12)

plt.plot(data=ts, subplots=True)


# another way of passing the x, y axis columns
import pandas as pd
df = pd.DataFrame({'month': [1, 4, 7, 10], 'sale': [55, 40, 84, 31]})
plt.plot('month','sale', data=df)

# multiple lines in a plot
df = pd.DataFrame({'x': [-1,2,3,4], 'y': [5,10,15,20],'z': [2,4,6,8]})
plt.plot(df.x,df[['y','z']]) # df.x is x and 2D passed on becomes values on y axis

# one more way
plt.plot(df.x,df.y,'g^', df.x, df.z,'b*')

# common axis plot
google = pd.read_csv('Google_Stock_Price_Train.csv',index_col='Date', parse_dates=['Date'])
google.head()
google.tail()
google['2012':'2016'].plot(subplots=True, figsize=(10,12))
plt.title('Google stock attributes from 2012 to 2016')
plt.show()

# seasonal subseries plot
# ref: https://queirozf.com/entries/pandas-time-series-examples-datetimeindex-periodindex-and-timedeltaindex

from statsmodels.graphics.tsaplots import month_plot
AirPassengers = pd.read_csv('airline-passengers.csv')
AirPassengers.plot()
# convert the column (it's a string) to datetime type
datetime_series = pd.to_datetime(AirPassengers['Month'])

# create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values)

df2=AirPassengers[['Passengers']].set_index(datetime_index)

month_plot(df2['Passengers'])
plt.show()


# Import data
df = pd.read_csv('airline-passengers.csv', parse_dates=['Month'])
x = df['Month'].values
y1 = df['Passengers'].values
import numpy as np
# Plot
fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')
plt.ylim(-800, 800)
plt.title('Air Passengers (Two Side View)', fontsize=16)
plt.hlines(y=0, xmin=np.min(df.Month), xmax=np.max(df.Month), linewidth=.5)
plt.show()


# Seasonal Plot of a Time Series

# Import Data
import matplotlib as mpl

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
df.reset_index(inplace=True)

# Prepare data
df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(10,5), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot('month', 'value', data=df.loc[df.year==y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)
plt.show()
