# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:31:38 2020

@author: Meekey
"""
# -*- coding: utf-8 -*-

import datetime as dt
import yfinance as yf
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import datetime as dt
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import seaborn as sns
from scipy import stats
# from bokeh.plotting import figure

# stocks =["^BSESN", "GTPL.BO", "TIDEWATER.NS", "SONATSOFTW.NS", "MANORG.BO", "SBIN.NS", "STRTECH.NS",
#           "GRANULES.NS", "M&MFIN.NS", "FSL.NS", "HINDUNILVR.NS", "TATAMOTORS.NS", "ITC.NS", "INDUSINDBK.NS", "LT.NS",
#           "MANAPPURAM.NS", "SPENCERS.NS", "SBICARD.NS", "RELIANCE.NS", "ASIANPAINT.NS", "BAJAJ-AUTO.NS","HCLTECH.NS",
#           "CAPLIPOINT.NS"
#         ]

# stocks = ['AXISBANK.NS', 'MCX.NS', 'SUDARSCHEM.NS','RAMCOCEM.NS', 'BANDHANBNK.NS',
#           'AMARAJABAT.NS', 'WIPRO.NS', 'BRITANNIA.NS', 'MOTHERSUMI.NS', 'BAJFINANCE.NS',
#           'CADILAHC.NS']




stocks = pd.read_excel("C:\\Users\\Meekey\\Documents\\DeepLearning\\All_Stocks3.xlsx")
# stock_beta  = pd.read_excel("C:\\Users\\Meekey\\Documents\\DeepLearning\\Beta.xlsx")
stocks = stocks.Symbol.tolist()

# stocks = ['VBL.NS']
rmse = {}
pred_price_list = {}
actual_price = {}
stk_beta = {}

start_date = "2010-01-01"
# train_end_dt = dt.datetime.today()- dt.timedelta(days=1)
train_end_dt = dt.datetime.today()

# train_end_dt = dt.datetime(2020,8,1) - dt.timedelta(days=1)
# test_date = dt.datetime.today()
test_date = dt.datetime.today()



pred_price_stock_list(stocks, start_date, train_end_dt, test_date, hdn_neurals = 150
                      , epcs = 70, btsize = 32, flag = 1)

def pred_price_stock_list(stocks, start_date, 
                          train_end_dt, test_date,
                          hdn_neurals = 50, epcs = 2, btsize = 32, flag = 0):
    
    
    ticker = stocks
    for ticker in stocks:
        print("Starting {} ===============================>".format(ticker))
        ohlcv = {}
        data = pd.DataFrame()
        ohlcv = pdr.get_data_yahoo(ticker, start_date, train_end_dt)
        # ohlcv['Adj Close'].plot()
        
        temp = ohlcv.iloc[len(ohlcv)-30:,]
        # temp['Adj Close'].plot(title = "last 30 days from training set")
        # ohlcv = ohlcv.drop(ohlcv.index[[0]])
        #save index return once for calculating beta
        # beta_df = pd.concat((beta_df, ohlcv['Adj Close']), axis = 0)
        # while index_flag == 1:
        #     index = ohlcv['Adj Close']
        #     index_return = index.pct_change().dropna()
        #     index_flag = 0
        # beta_flag = 1
        # if index_flag == 0 and beta_flag == 1:
        #     stock_return = ohlcv['Adj Close'].pct_change().dropna()
        #     stock_beta, stk_alpha = stats.linregress(index_return, 
        #                 stock_return)[0:2]
        #     stk_beta[ticker] = stock_beta

    
    #-----------------------------------------------------------------------------
    #MACD
    
        def MACD(DF,a,b,c):
            df = DF.copy()
            # df = ohlcv.copy()
            df['MA_fast'] = df['Adj Close'].ewm(span = a, min_periods = a).mean()
            df['MA_slow'] = df['Adj Close'].ewm(span = b, min_periods = b).mean()
            df['MACD'] = df['MA_fast'] - df['MA_slow']
            df['Signal'] = df['MACD'].ewm(span = c, min_periods = c).mean()
            #df = df.dropna()
            df = df.iloc[:,[8,9]]
            return df
        
        #------------------------------------------------------------------------------
        # key_stats = pd.DataFrame()
        # MACD(ohlcv,12,26,9)
        data = MACD(ohlcv,12,26,9)
        key_stats = data.tail(10).copy()
        if flag == 0:
            plt.plot(data['MACD'], color = 'purple', label = 'MACD')
            plt.plot(data['Signal'], color = 'red', label = 'Signal')
            plt.title("{} - All train data".format(ticker))
            plt.legend()
            plt.xticks(rotation=45)
            plt.show()
        
        macd_30_days = MACD(temp,12,26,9)
        if flag == 0:
            plt.plot(macd_120_days['MACD'], color = 'purple', label = 'MACD')
            plt.plot(macd_120_days['Signal'], color = 'red', label = 'Signal')
            plt.title("{} - Last 30 days MACD".format(ticker))
            plt.legend()
            plt.xticks(rotation=45)
            plt.show()
        #------------------------------------------------------------------------------
        #ATR
        
        def ATR(DF,n):
            df = DF.copy()
            df['H-L'] = abs(df['High'] - df['Low'])
            df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
            df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
            df['TR'] = df[['H-L','H-PC','L-PC']].max(axis = 1, skipna = False)
            df['ATR'] = df['TR'].rolling(n).mean()
            df2 = df.drop(['H-L','H-PC','L-PC'], axis = 1)
            #df2 = df2.dropna()
            return df2
        
        df_atr = ATR(ohlcv,20).iloc[:,[6,7]]
        data['TR'] = df_atr.iloc[:,0].values
        data['ATR'] = df_atr['ATR'].values
        if flag == 0:
            plt.plot(data['TR'], color = 'purple', label = 'TR')
            plt.plot(data['ATR'], color = 'green', label = 'ATR')
            plt.title("{} - All train data".format(ticker))
            plt.xticks(rotation=45)
            plt.legend()
            plt.show()
        # key_stats = data.tail(20)
        
        
        atr_120_days = ATR(temp,20).iloc[:,[6,7]]
        if flag == 0:
            plt.plot(atr_120_days['TR'], color = 'purple', label = 'TR')
            plt.plot(atr_120_days['ATR'], color = 'red', label = 'ATR')
            plt.title("{} - Last 120 days ATR".format(ticker))
            plt.legend()
            plt.xticks(rotation=45)
            plt.show()
        #------------------------------------------------------------------------------
        #df = ohlcv.copy()
        #n = 20
        def BollBnd(DF,n):
            df = DF.copy()
            df['MA20Days'] = df['Adj Close'].rolling(n).mean()
            df['BB_up'] = df['MA20Days'] + 2 * df['MA20Days'].rolling(n).std()
            df['BB_down'] = df['MA20Days'] - 2 * df['MA20Days'].rolling(n).std()
        #    df['MA'] = df['Adj Close'].ewm(span = n, min_periods = n).mean()
        #    df['BB_up'] = df['MA'] + df['MA'].ewm(span = n, min_periods = n).std()
        #    df['BB_down'] = df['MA'] - df['MA'].ewm(span = n, min_periods = n).std()
        
            df['BB_range'] = df['BB_up'] - df['BB_down']
            df2 = df.iloc[:,[6,7,8]]
            #df2 = df2.dropna()
            return df2
        
        BB_train = BollBnd(ohlcv, 20)
        data['MA20Days'] = BB_train['MA20Days']
        data['BB_up'] = BB_train['BB_up']
        data['BB_down'] = BB_train['BB_down']
        if flag == 0:
            plt.plot(data['MA20Days'], color = 'purple', label = 'MA20Days')
            plt.plot(data['BB_up'], color = 'green', label = 'BB_up')
            plt.plot(data['BB_down'], color = 'blue', label = 'BB_down')
            plt.xticks(rotation=45)
            plt.title("{} - BB Train Data".format(ticker))
            plt.legend()
            plt.show()
        
        BB_120_days = BollBnd(temp, 20)
        if flag == 0:
            plt.plot(BB_120_days['MA20Days'], color = 'purple', label = 'MA20Days')
            plt.plot(BB_120_days['BB_up'], color = 'green', label = 'BB_up')
            plt.plot(BB_120_days['BB_down'], color = 'blue', label = 'BB_down')
            plt.xticks(rotation=45)
            plt.title("{} - BB Last 120 days".format(ticker))
            plt.legend()
            plt.show()
        
        # DF = ohlcv.copy()
        # n = 14
        def RSI(DF,n):
            df = DF.copy()
            df['delta'] = df['Adj Close'] - df['Adj Close'].shift(1)
            df['gain'] = np.where(df['delta']>=0, df['delta'],0)
            df['loss'] = np.where(df['delta']<0,abs(df['delta']),0)
            avg_gain = []
            avg_loss = []
            gain = df['gain'].tolist()
            loss = df['loss'].tolist()
            for i in range(len(df)):
                if i < n:
                    avg_gain.append(np.NaN)
                    avg_loss.append(np.NaN)
                elif i == n:
                    avg_gain.append(df['gain'].rolling(n).mean()[n].tolist())
                    avg_loss.append(df['loss'].rolling(n).mean()[n].tolist())
                elif i > n:
                    avg_gain.append(((n-1) * avg_gain[i-1] + gain[i])/n)
                    avg_loss.append(((n-1) * avg_loss[i-1] + loss[i])/n)
            df['avg_gain'] = np.array(avg_gain)
            df['avg_loss'] = np.array(avg_loss)
            df['RS'] = df['avg_gain']/df['avg_loss']
            df['RSI'] = 100 - (100/(1+df['RS']))
            return df['RSI']
        
        data['RSI'] = RSI(ohlcv, 14)
        data['label'] = ohlcv['Adj Close']
        from sklearn.preprocessing import MinMaxScaler
        plt_sc = MinMaxScaler()
        rsi_price = data.iloc[:,7:]
        # rsi_price = rsi_price.dropna()
        rsi_price = pd.DataFrame(plt_sc.fit_transform(rsi_price),
                                 columns = ['RSI', 'Stock Price'])
        rsi_price_ind = pd.DataFrame(rsi_price)
        rsi_price_ind.index = data.index
        if flag == 0:
            plt.plot(rsi_price['RSI'], color = 'purple', label = 'RSI')
            plt.plot(rsi_price['Stock Price'], color = 'green', label = 'Stock Price')
            plt.xticks(rotation=45)
            plt.title("{} - RSI vs Stock Price".format(ticker))
            plt.legend()
            plt.show()
        
        
        def ADX(DF,n):
            "function to calculate ADX"
            df2 = DF.copy()
            df2['TR'] = ATR(df2,n)['TR'] #the period parameter of ATR function does not matter because period does not influence TR calculation
            df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
            df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
            df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
            df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
            TRn = []
            DMplusN = []
            DMminusN = []
            TR = df2['TR'].tolist()
            DMplus = df2['DMplus'].tolist()
            DMminus = df2['DMminus'].tolist()
            for i in range(len(df2)):
                if i < n:
                    TRn.append(np.NaN)
                    DMplusN.append(np.NaN)
                    DMminusN.append(np.NaN)
                elif i == n:
                    TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
                    DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
                    DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
                elif i > n:
                    TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
                    DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
                    DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
            df2['TRn'] = np.array(TRn)
            df2['DMplusN'] = np.array(DMplusN)
            df2['DMminusN'] = np.array(DMminusN)
            df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
            df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
            df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
            df2['DIsum']=df2['DIplusN']+df2['DIminusN']
            df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
            ADX = []
            DX = df2['DX'].tolist()
            for j in range(len(df2)):
                if j < 2*n-1:
                    ADX.append(np.NaN)
                elif j == 2*n-1:
                    ADX.append(df2['DX'][j-n+1:j+1].mean())
                elif j > 2*n-1:
                    ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
            df2['ADX']=np.array(ADX)
            return df2['ADX']
    
        data = data.drop(['label'], axis = 1)
        data['ADX'] = ADX(ohlcv, 14)
        data['label'] = ohlcv['Adj Close']
        # adx_price = data.iloc[:,8:]
        adx_price = data['ADX']
        adx_price = pd.concat((adx_price, data['label']), axis = 1)
        adx_price = pd.DataFrame(plt_sc.fit_transform(adx_price),
                                 columns = ['ADX', 'Stock Price'])
        adx_price_ind = pd.DataFrame(adx_price).copy()
        adx_price_ind.index = data.index
        if flag == 0:
            plt.plot(adx_price['ADX'], color = 'purple', label = 'ADX')
            plt.plot(adx_price['Stock Price'], color = 'green', label = 'Stock Price')
            plt.xticks(rotation=45)
            plt.title("{} - ADX".format(ticker))
            plt.legend()
            plt.show()
            
        data_temp = data.copy()
        data_temp = data_temp.dropna()
        data_temp.to_csv("C:\\Users\\Meekey\\Documents\\CAPM\\Spyder Code\\Flask\\VBL.csv")
        #%%
        #Building the RNN
        # def RNN_Model1(data_for_rnn, neurals, ep, bs):
        temp = data.dropna()
        data_for_rnn = temp.values
        tech_ind = pd.DataFrame()
        tech_ind = pd.DataFrame(data_for_rnn)
        tech_ind = tech_ind.dropna()
        train = tech_ind.iloc[:len(tech_ind)-30,1:]
        test = tech_ind.iloc[len(tech_ind)-30:,1:]
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
        # hdn_neurals = 50
        # epcs = 2
        # btsize = 32
        #Building the RNN
        #importing keras and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        #initializing RNN
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
        fit_rnn = regressor.fit(x_train, y_train, epochs = epcs, batch_size = btsize)
        # real_stock_price = test.iloc[:,-1].values
        # real_stock_price = real_stock_price.reshape(-1,1)
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
        pred_price = regressor.predict(x_test)
        pred_price = sc.inverse_transform(pred_price)
        real_stock_price = pd.DataFrame(test.iloc[:,-1].values, columns = ['Real Stock Price'])
        real_stock_price['Predicted Price'] = pred_price
        from sklearn.metrics import mean_squared_error
        mod1_mse = mean_squared_error(real_stock_price['Real Stock Price'], pred_price)
        rmse[ticker] = np.sqrt(mod1_mse)
        # indexed_table = pd.DataFrame()
        # indexed_table = pd.DataFrame(pd.concat((real_stock_price, pred_price), axis = 1),
                                     # columns = ['Real Price', 'Predicted Price']
        # plt.plot(indexed_table['Real Price'], color = 'red')
        # plt.plot(indexed_table['Predicted Price'], color = 'blue')
    
        plt.plot(real_stock_price['Real Stock Price'], color = 'red', label = 'real stock price')
        plt.plot(real_stock_price['Predicted Price'], color = 'blue', label = 'pred stock price')
        plt.title('{}, EPOCHS: {}'.format(ticker,epcs))
        plt.title(ticker)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        # pred_next_day_price = regressor.predict(x_test[[-1]])
        # pred_next_day_price = sc.inverse_transform(pred_next_day_price)
        # pred_price_list[ticker] = pred_next_day_price
        # last_price = yf.download(ticker, test_date, test_date)['Adj Close']
        # actual_price[ticker] = last_price.values
    
    
    
    
    
    
    
    
    
    
    
    
    
