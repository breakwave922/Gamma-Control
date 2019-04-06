import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense, Activation
import time
import csv
from sklearn.metrics import mean_squared_error
from scipy import signal
import pdb
from utils import series_to_supervised, fit_lstm, forecast_lstm, make_forecasts

np.random.seed(12345)

def butter_bandpass(lowcut, highcut, fs, order=2):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = signal.butter(order, [low, high], btype='band')
	return b, a

################## Load data ###################

df = pd.read_csv('./LFP_elec_0',header=None)
LFP = df.values

fs = 1000

time = np.linspace(1/fs,LFP.shape[0]/fs,LFP.shape[0])

# Filter 

b_filt, a_filt = butter_bandpass(lowcut = 50, highcut = 70, fs = 1000)

LFP_filt = signal.filtfilt(b_filt,a_filt,LFP[:,0:1],axis=0)

# Plot time domain and PSD
f, Pxx_den = signal.welch(LFP[:,0],fs,nperseg=2001)

plt.figure()
plt.subplot(2,1,1)
plt.plot(time,LFP[:,0])
plt.plot(time,LFP_filt)
plt.xlabel('time(ms)')
plt.subplot(2,1,2)
plt.semilogy(f,Pxx_den)
plt.xlabel('frequency (hz)')
plt.show()

############# Process data for Neural Network ##############

# Series to supervised
num_lookback = 100
num_predict = 15

# n_in is the number of samples for "input" and n_out is the number for "output", or prediction
supervised_dataset = series_to_supervised(LFP_filt[:,0:1],n_in=num_lookback,n_out=num_predict)
sup_ds_filt = supervised_dataset.values

supervised_dataset = series_to_supervised(LFP[:,0:1],n_in=num_lookback,n_out=num_predict)
sup_ds_raw = supervised_dataset.values


# train    
model_lstm = fit_lstm(sup_ds_filt, num_predict, 10, nb_epoch = 1, n_neurons=1000)
model_lstm.summary()

# Make forecasts on test data
scaled_forecasts = make_forecasts(model_lstm, 1, sup_ds_filt, num_predict)

# Make persistence forecasts
y_pers_test = np.transpose(np.tile(sup_ds_filt[:,-num_predict-1], (num_predict,1)))

# Calculate root mean squared error on forecasts
rmse_lstm_test = np.sqrt(np.mean((scaled_forecasts-sup_ds_filt[:,-num_predict:])**2,axis=0))
print("rmse for (LSTM) testing: ", rmse_lstm_test)

rmse_pers_test = np.sqrt(np.mean((y_pers_test-sup_ds_filt[:,-num_predict:])**2,axis=0))
print("rmse for (persistence) testing: ", rmse_pers_test)

plt.figure()
plt.bar(np.arange(0,num_predict)-0.2,rmse_pers_test,0.3,label='rmse of persistence forecast')
plt.bar(np.arange(0,num_predict)+0.2,rmse_lstm_test,0.3,label='rmse of lstm forecast')
plt.legend()

#look at 6 samples to see how we did
chunk_size = num_lookback + num_predict

plt.figure(figsize=(20,10))
for i in np.arange(1,7):
    sample = np.random.randint(0,scaled_forecasts.shape[0])
    plt.subplot(2,3,i)
    plt.plot(np.linspace(1,chunk_size,num=chunk_size),sup_ds_filt[sample,:],color='blue',label='actual')
    plt.plot(np.linspace(num_lookback+1,chunk_size,num=num_predict),scaled_forecasts[sample,:],color='red',label='predicted')

############ "Real Time" LFP #################

sup_ds_rt_filt = signal.filtfilt(b_filt,a_filt,sup_ds_raw,axis=1)

# Make real-time forecasts

# Forecast
rt_forecasts = make_forecasts(model_lstm, 1, sup_ds_rt_filt, num_predict)


# Plot
plt.figure(figsize=(20,10))
for i in np.arange(1,7):
    sample = np.random.randint(0,sup_ds_rt_filt.shape[0])
    plt.subplot(2,3,i)
    plt.plot(np.linspace(1,chunk_size,num=chunk_size),sup_ds_filt[sample,:],color='blue',label='actual')
    plt.plot(np.linspace(1,chunk_size,num=chunk_size),sup_ds_rt_filt[sample,:],color='orange',label='actual')
    plt.plot(np.linspace(num_lookback+1,chunk_size,num=num_predict),rt_forecasts[sample,:],color='red',label='predicted')


plt.show()


