import numpy as np
import pandas as pd
import seaborn as sns
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

# def plot(real, pred, trainlen):
#     future = 100
#     plt.figure(figsize=(18,8))
#     plt.plot(range(0,trainlen+future),real[0:trainlen+future],'k',label="target system")
#     plt.plot(range(trainlen,trainlen+100),pred.reshape(-1,1),'r', label="free running ESN")
#     lo,hi = plt.ylim()
#     plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
#     plt.legend(loc=(0.61,0.12),fontsize='x-large')
#     sns.despine();