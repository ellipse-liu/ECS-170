import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import os
import tensorflow as tf
import yfinance as yf
from pyESN import ESN
import matplotlib.pyplot as plt

def grab_data(symbol, start_date, end_date, intervals):
    stock_data = yf.download(symbol, start=start_date, end=end_date, interval=intervals ,progress=False)
    data = stock_data["Close"].values
    return data
    
def MSE(prediction, actual):
    return np.mean(np.power(np.subtract(np.array(prediction),actual),2))


def run_echo(data, reservoir_size=500, sr=1.2, n=0.005, window=5):
    esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = reservoir_size,
          sparsity=0.2,
          random_state=23,
          spectral_radius=sr,
          noise = n)

    trainlen = 100
    current_set = []
    for i in range(0,100):
        pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
        prediction = esn.predict(np.ones(window))
        current_set.append(prediction[0])
    current_set = np.reshape(np.array(current_set),(-1,100))
    mse = MSE(current_set, data[trainlen:trainlen+100])
    
    return (mse, current_set)

def future_pred(data, reservoir_size=500, sr=1.2, n=0.005, window=5):
    esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = reservoir_size,
          sparsity=0.2,
          random_state=23,
          spectral_radius=sr,
          noise = n)
    pred_training = esn.fit(np.ones(100),data[-100:])
    prediction = esn.predict(np.ones(window))
    return prediction.reshape(-1)
    
def main():
    st.title("Stock Price Prediction App")

    st.write("Enter the stock symbol and dates for prediction:")

    stock_symbol = st.text_input("Stock Symbol", value='AAPL', max_chars=5)
    start_date = st.text_input("Start Date", value='2023-01-01')
    end_date = st.text_input("End Date", value='2023-01-31')

    predict_button = st.button("Predict")

    if predict_button:
        # Perform your prediction logic here
        st.write(f"Predicting for stock symbol: {stock_symbol}")
        st.write(f"Start date: {start_date}")
        st.write(f"End date: {end_date}")
        data = grab_data(stock_symbol, start_date, end_date, '1d')
        prediction = future_pred(data, 500, 1.2, .005, 5)
        st.write(prediction)
        # You can call your prediction function and display the results here

if __name__ == "__main__":
    main()