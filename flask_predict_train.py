# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:30:50 2020

@author: Meekey
"""
import pandas as pd
# import datetime as dt
# import yfinance as yf
import pandas as pd
# import pandas_datareader.data as pdr
import numpy as np
# import datetime as dt
# import csv
import os
os.getcwd()
os.chdir("/Users/meekey/Documents/GitHub/Flask")

data = pd.read_excel("tech_ind.xlsx")
train = data.iloc[:len(data)-30,1:]
test = data.iloc[len(data)-30:,1:]
x_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values
y_train = y_train.reshape(-1,1)
x_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values
y_test = y_test.reshape(-1,1)


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#%%
hdn_neurals = 50
epcs = 3
btsize = 32
#Building the RNN
#importing keras and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = hdn_neurals, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = hdn_neurals, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = hdn_neurals, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = hdn_neurals, return_sequences = False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer="adam", loss = "mean_squared_error")

regressor.fit(x_train, y_train, epochs = epcs, batch_size = btsize)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
pred_price = regressor.predict(x_test)
pred_price = sc.inverse_transform(pred_price)
real_stock_price = pd.DataFrame(test.iloc[:,-1].values, columns = ['Real Stock Price'])
real_stock_price['Predicted Price'] = pred_price
from sklearn.metrics import mean_squared_error
mod1_mse = mean_squared_error(real_stock_price['Real Stock Price'], pred_price)
rmse = np.sqrt(mod1_mse)
print(rmse)

import pickle
with open('regressor.pkl', 'wb') as model_pkl:
    pickle.dump(regressor,model_pkl)




























