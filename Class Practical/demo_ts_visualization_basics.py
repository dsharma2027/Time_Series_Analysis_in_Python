# source: machine learning mastery

from pandas import read_csv
from matplotlib import pyplot as plt
import os
from pandas import DataFrame
from pandas import Grouper
os.chdir('D:\\ssridhar\\research\\AnalyticsBigdataML\\ML Course\\Praxis\\July 2020\\TSF\\Visualizing Time Series data')

print("This is the cwd: ", os.getcwd())

series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
print(series.head())

# line plot
series.plot()
plt.show()

# Or scatter plot
series.plot(style='k.')
plt.show()

# histogram
series.hist()
plt.show()

# density plot
series.plot(kind='kde')
plt.show()

# seasonal plots
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.plot(subplots=True, legend=False)
plt.show()

# boxplot by year
years.boxplot()
plt.show()


# create a boxplot of monthly data
from pandas import concat
one_year = series['1990']
groups = one_year.groupby(Grouper(freq='M'))
months = concat([DataFrame(x[1].values) for x in groups], axis=1)
months = DataFrame(months)
months.columns = range(1,13)
months.boxplot()
plt.show()

# lag plot
from pandas.plotting import lag_plot
lag_plot(series)
plt.show()

# plot against more lags
values = DataFrame(series.values)
lags = 7
columns = [values]
for i in range(1,(lags + 1)):
	columns.append(values.shift(i))
dataframe = concat(columns, axis=1)
columns = ['t+1']
for i in range(1,(lags + 1)):
	columns.append('t-' + str(i))
dataframe.columns = columns
plt.figure(1)
for i in range(1,(lags + 1)):
	ax = plt.subplot(240 + i)
	ax.set_title('t+1 vs t-' + str(i))
	plt.scatter(x=dataframe['t+1'].values, y=dataframe['t-'+str(i)].values)
plt.show()

