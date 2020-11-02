from keras.models import load_model
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import linregress
from sklearn.utils import shuffle
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.dates as mdates
import datetime
from datetime import date
import schedule
import keras
import time

#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

print('Program started')
time.sleep(60)
def main_func():

    now = datetime.datetime.now()
    year = now.year
    week = now.isocalendar()[1]
    df = pd.read_csv('data_'+str(year)+'W'+str(week)+'.csv', engine='python' )

    dim=df.shape[0]
    #samples in one week (60x24x7 = 10 080)
    t_train=10000 #Here you can change the amount of data you want to start the training.

    if dim < t_train:
        print("Need more data to start training...")
        print_func()

    if dim >= t_train:
        print("Enough data collected for start training...")
        train_func()

def print_func():

    print("Printing data...")
    now = datetime.datetime.now()
    year = now.year
    week = now.isocalendar()[1]
    df = pd.read_csv('data_'+str(year)+'W'+str(week)+'.csv', engine='python' )
    print("raw shape:")
    print (df.shape)
    dim=df.shape[0]

    df.head()
    features_considered  = ['Active power consumed [W]','Voltage AC [V_ac]','Current AC [I_ac]']
    features = df[features_considered]
    features.index = df['Time stamp']
    features.head()
    features.plot(subplots=True, legend=False)
    [ax.legend(loc=1) for ax in plt.gcf().axes]
    plt.show(block=False)
    plt.pause(60)
    plt.close()

def train_func():

    today = date.today()
    now = datetime.datetime.now()
    year = now.year
    week = now.isocalendar()[1]
    print("Starting training...")
    fi = 'data_'+str(year)+'W'+str(week)+'.csv'
    raw = pd.read_csv(fi, delimiter=',', engine='python' )
    raw = raw.drop('Time stamp', axis=1)

    print("raw shape:")
    print (raw.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    raw = scaler.fit_transform(raw)

    time_shift = 1 #shift is the number of steps we are predicting ahead
    n_rows = raw.shape[0] #n_rows is the number of time steps of our sequence
    n_feats = raw.shape[1]
    train_size = int(n_rows * 0.8)

    train_data = raw[:train_size, :] #first train_size steps, all 5 features
    test_data = raw[train_size:, :] #I'll use the beginning of the data as state adjuster

    x_train = train_data[:-time_shift, :] #the entire train data, except the last shift steps
    x_test = test_data[:-time_shift,:] #the entire test data, except the last shift steps
    x_predict = raw[:-time_shift,:] #the entire raw data, except the last shift steps

    y_train = train_data[time_shift:, :]
    y_test = test_data[time_shift:,:]
    y_predict_true = raw[time_shift:,:]

    x_train = x_train.reshape(1, x_train.shape[0], x_train.shape[1]) #ok shape (1,steps,5) - 1 sequence, many steps, 5 features
    y_train = y_train.reshape(1, y_train.shape[0], y_train.shape[1])
    x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])
    y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])
    x_predict = x_predict.reshape(1, x_predict.shape[0], x_predict.shape[1])
    y_predict_true = y_predict_true.reshape(1, y_predict_true.shape[0], y_predict_true.shape[1])

    print("\nx_train:")
    print (x_train.shape)
    print("y_train")
    print (y_train.shape)
    print("x_test")
    print (x_test.shape)
    print("y_test")
    print (y_test.shape)

    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2, strides=2))
    #model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2, strides=2))
    #model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(256, return_sequences=True))
    #model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(n_feats, return_sequences=True))

    model.add(Dropout(0.1))
    #model.add(Flatten())
    #model.add(Dense(n_feats))
    time_callback = TimeHistory()
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=600, batch_size=16, verbose=2)
    times = time_callback.times
    model.save('NN_'+str(today)+'W'+str(week)+'.h5')
    print("Training finished")
    model.summary()

    model_A = tf.keras.models.load_model('NN_'+str(today)+'.h5')
    y_predict_model = model_A.predict(x_predict)
    y_predict_model2 = model_A.predict(x_test)
    y_predict_model3 = model_A.predict(x_train)


    print("\ny_predict_true:")
    print (y_predict_true.shape)
    print("y_predict_model_global: ")
    print (y_predict_model.shape)
    print("y_predict_model_validation: ")
    print (y_predict_model2.shape)
    print("y_predict_model_train: ")
    print (y_predict_model3.shape)

    test_size = n_rows - train_size
    print("test length: " + str(test_size))

    #print("-------------------------------MSE------------------------------------------------")
    mse = np.square(np.subtract(y_predict_true,y_predict_model)).mean()
    mse2 = np.square(np.subtract(y_test,y_predict_model2)).mean()
    mse3 = np.square(np.subtract(y_train,y_predict_model3)).mean()
    #print("-------------------------------RMSE---------------------------------------------")
    rmse = np.sqrt(mse)
    rmse2 = np.sqrt(mse2)
    rmse3 = np.sqrt(mse3)
    #print("-------------------------------MAE------------------------------------------------")
    mae = np.abs(np.subtract(y_predict_true,y_predict_model)).mean()
    mae2 = np.abs(np.subtract(y_test,y_predict_model2)).mean()
    mae3 = np.abs(np.subtract(y_train,y_predict_model3)).mean()
    print("--------------------------------MSE-----------------------------------------------")
    print("MSE metrics for CNN_LSTM_5 model:")
    print("MSE validation: " + str(mse2))
    print("MSE train: " + str(mse3))
    print("MSE global: " + str(mse))

    print("--------------------------------RMSE-----------------------------------------------")
    print("RMSE metrics for CNN_LSTM_5 model:")
    print("RMSE validation: " + str(rmse2))
    print("RMSE train: " + str(rmse3))
    print("RMSE global: " + str(rmse))
    print("--------------------------------MAE-----------------------------------------------")
    print("MAE metrics for CNN_LSTM_5 model:")
    print("MAE validation: " + str(mae2))
    print("MAE train: " + str(mae3))
    print("MAE global: " + str(mae))

    predict_plot = scaler.inverse_transform(y_predict_model[0])
    true_plot = scaler.inverse_transform(y_predict_true[0])

    predict_plot = predict_plot[:,0]
    true_plot = true_plot[:,0]

    plt.figure(figsize=(16,6))

    plt.plot(true_plot, label='True',linewidth=1)
    plt.plot(predict_plot,  label='CNN_LSTM_5',color='y',linewidth=1)
    if train_size > 0:
        maxVal = max(true_plot.max(),predict_plot.max())
        minVal = min(true_plot.min(),predict_plot.min())

        plt.plot([train_size,train_size],[minVal,maxVal],label='train/test limit',color='k')

    plt.ylabel('Active power consumed [W]')
    plt.xlabel('Time [/min]')
    plt.legend()
    plt.show(block=False)
    plt.pause(30)
    plt.close()

schedule.every(1).seconds.do(main_func)
while True:
    schedule.run_pending()
