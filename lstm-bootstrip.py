# -*- coding: utf-8 -*-

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

scaler = MinMaxScaler(feature_range=(0, 1))

def get_dataframe(data_path):
    return pd.read_csv(data_path)

#def dataproc(dataframe):
#    dataframe = dataframe[['Store', 'Dept', 'Date', 'Weekly_Sales']]
#    return dataframe
    
#def create_dataset(dataframe, lookback):
#    dataX, dataY = [], []
#    for i in range(len(dataframe) - lookback - 1):
#        dataX.append(dataframe[i:(i+lookback)])
#        dataY.append(dataframe[i+lookback])
#    return np.array(dataX), np.array(dataY)

def create_dataset(dataframe, lookback):
    dataX, dataY = [], []
    for i in range(len(dataframe) - lookback - 1):
        dataX.append(dataframe[i:(i+lookback), 0])
        dataY.append(dataframe[i+lookback, 0])
    return np.array(dataX), np.array(dataY)

def normalize_data(dataset):
    dataset = dataset.values.reshape(-1, 1)
    dataset = scaler.fit_transform(dataset)
    return dataset

def create_lstm_model(trainX,trainY,look_back):
    model = Sequential()
    model.add(LSTM(16, input_shape=(1, look_back), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=150, batch_size=2, verbose=1)
    return model

def calcu_score(PredictY, groundtruth):
    weights = groundtruth[1]
    for i in range(len(weights)):
        if weights[i] == 0:
            weights[i] = 1
        else:
             weights[i] = 5
    score = sum(abs(PredictY-groundtruth[0])*weights)/sum(weights)
    print('Test Score: %f mae:',score)
    
def split_data(dataset):
    train_size = int(len(dataset) * 0.7)
    return dataset[0:train_size], dataset[train_size:len(dataset)]

if __name__ == '__main__':
    trainframe = get_dataframe('train.csv')
    #the following are test codes
    traintest = trainframe[trainframe.Store == 1]
    traintest = traintest[traintest.Dept == 3]
    train, test = split_data(traintest)
    
    lookback = 1
    train_sales = train.Weekly_Sales
    train_sales = normalize_data(train_sales)
    trainX,trainY = create_dataset(train_sales, lookback)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    model = create_lstm_model(trainX,trainY,lookback)
    testY = []
    groundtruth = [test.Weekly_Sales.values, test.IsHoliday.values.astype(int)]
    for ind in test.index:
#        testX = []
#        testX.append(train.Weekly_Sales[ind-51-3:ind-51])
        testX = train.Weekly_Sales[ind-51-1-143*2:ind-51-143*2]
#        testX = {'sales':testX}
#        testX = pd.DataFrame(testX)
        testX = normalize_data(testX)
        testX = np.transpose(testX)
        testX = np.reshape(testX, (testX.shape[0],1,testX.shape[1]))
        testpredict = model.predict(testX)
        testpredict = scaler.inverse_transform(testpredict)
        testY.append(testpredict[0][0])
    calcu_score(np.array(testY),groundtruth)
    
    
    