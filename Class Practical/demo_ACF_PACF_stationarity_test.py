import os
os.getcwd()
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller

df_stk=pd.read_csv("stock_price.csv")
df_stk.head()

y = df_stk['price']

# compare values with working in Excel
acf(y,unbiased = True, nlags=2)
pacf(y,method='ols', nlags=10)



# one more demo
df_qk=pd.read_csv("earthquakes.csv")
df_qk.head()

y = df_qk['Quakes']

# compare values with working in Excel
acf(y,unbiased = True, nlags=2)
pacf(y,method='ols', nlags=10)


# check for stationarity - Augmented Dickey Fuller test
def check_stationarity(y):    
    result = adfuller(y)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# try this test for both datasets and see the -ve and +ve results
y = df_stk['price']
check_stationarity(y)
y = df_qk['Quakes']
check_stationarity(y)

# try differencing for stock price dataset

df_stk['price_diff1'] = df_stk['price'] - df_stk['price'].shift(1)
df_stk.head()
check_stationarity(df_stk.price_diff1[1:])


# confidence intervals
(acfs, acf_CI) = acf(y,unbiased = True, nlags = 2, alpha=0.05)
acf_CI
acfs
