# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:20:23 2021

@author: guoyu
"""
import os
os.chdir('C:/stock')

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from data_scale import data_scale
from tensorflow.keras.layers import Input, LSTM, Dense,Dropout
from tensorflow.keras import Model


stock = yf.download('SPY', 
                      start='2021-01-01', 
                      end='2021-12-27', 
                      progress=False, interval = '1h')

######stock up or down prediction
stock['change'] = stock.Close - stock.Open
stock['per_change'] = 100 * stock.change/stock.Open
stock['res'] = 1
stock.res[stock.change < 0] = 0
stock = stock.drop('change', axis = 1)
stock = stock.drop('Adj Close', axis = 1)

feature = stock.shape[1]
time_step = 12

scaler = MinMaxScaler(feature_range=(0, 1))
X_train, y_train, X_test, y_test = data_scale.load_data(stock, feature, time_step, scaler)
y_train, y_test = y_train[:,-1], y_test[:,-1]

input_val = Input(shape = (time_step, feature))  
lstm_1 = LSTM(8, return_sequences = True)(input_val)    
lstm_2 = LSTM(4)(lstm_1)

dense_1 = Dense(4)(lstm_2) 
dense_2 = Dense(1, activation = 'sigmoid')(dense_1) 

model = Model(input_val, dense_2)
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 32, verbose = 1)

y = model.predict(X_test)




######stock price prediction













######stock feature preparation