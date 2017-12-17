# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:11:55 2017

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
dataframe = pd.read_csv('train.csv', index_col = 'Date', date_parser = dateparse)
dataset = dataframe[dataframe.Store == 1]
dataset = dataset[dataset.Dept == 1]
dataset = dataset[['Weekly_Sales']]

#dataset = dataset.astype('float32')
#
#plt.plot(dataset)
#plt.show()

#print(dataset)

mod = sm.tsa.statespace.SARIMAX(dataset,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

pred = results.get_prediction(start=pd.to_datetime('2012-04-06'), dynamic=False)
pred_ci = pred.conf_int()

ax = dataset['2012-04-06':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Weekly Sales')
plt.legend()

plt.show()

y_forecasted = pred.predicted_mean
y_forecasted = np.array(y_forecasted['2012-04-06':])
y_truth = np.array(dataset['2012-04-06':])

# Compute the mean square error
result = y_forecasted - y_truth
mse = (result[0,:]).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))