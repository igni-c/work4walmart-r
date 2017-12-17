# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:33:26 2017

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

dataframe = pd.read_csv('train.csv')
dates = dataframe.Date[dataframe.Store == 1]
#dates = dates[dataframe.Dept == 1].tolist()


testdata = dataframe[dataframe.Store == 1]
testdata = testdata.Weekly_Sales[dataframe.Dept == 3]
#testdata = testdata.astype('float32')

plt.plot(testdata)
plt.show()

def create_dataset(dataset, lookback):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback - 1):
        dataX.append(dataset[i:(i+lookback), 0])
        dataY.append(dataset[i+lookback, 0])
    return np.array(dataX), np.array(dataY)

#np.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
testdata = testdata.values.reshape(-1, 1)
testdata = scaler.fit_transform(testdata)


# split into train and test sets
train_size = int(len(testdata) * 0.6)
test_size = len(testdata) - train_size
train, test = testdata[0:train_size,:], testdata[train_size:len(testdata),:]

# use this function to prepare the train and test datasets for modeling
look_back = 7
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(16, input_shape=(1, look_back), activation='relu',return_sequences=True))
model.add(LSTM(16, input_shape=(1, look_back), activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainX, trainY, epochs=150, batch_size=2, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f mae' % (trainScore))
testScore = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f mae' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(testdata)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(testdata)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(testdata)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(testdata))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
