# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:52:15 2020

@author: Meekey
"""
import pickle
from flask import Flask, request
import numpy as np
import pandas as pd
import os
import tensorflow as tf

global graph
graph = tf.get_default_graph()
    
os.chdir("/Users/meekey/Documents/GitHub/Flask")
with open('regressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
app = Flask(__name__)
@app.route('/predict_file', methods=["GET", "POST"])
def pred_price():
    input_data = pd.read_csv(request.files.get("input_file"))
    # input_data = pd.read_csv("input.csv")
    input_data = input_data.drop(["Date"], axis = 1)
    x_test = input_data.iloc[:,:-1]
    x_test = x_test.values
    y_test = input_data.iloc[:,-1]
    y_test = y_test.values
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    x_test = sc.fit_transform(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = y_test.reshape(-1,1)
    y_test = sc.fit_transform(y_test)
    with graph.as_default():
	    prediction = model.predict(x_test)
    prediction = sc.inverse_transform(prediction)
    return str(prediction)
if __name__ == '__main__':
    app.run()







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    