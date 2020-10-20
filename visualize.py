from keras.models import load_model
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import linregress
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.dates as mdates
import datetime

today = "data_2020W45"
print("Opening data...")
fi = 'data_'+str(today)+'.csv'
raw = pd.read_csv(fi, delimiter=',', engine='python' )
raw = raw.drop('Time stamp', axis=1)

print("raw shape:")
print (raw.shape)



def plot(true, predicted, divider):

    predict_plot = scaler.inverse_transform(predicted[0])
    true_plot = scaler.inverse_transform(true[0])

    predict_plot = predict_plot[:,0]
    true_plot = true_plot[:,0]

    plt.figure(figsize=(16,6))

    plt.plot(true_plot, label='True',linewidth=1)
    #plt.plot(true_plot, label='True PVPG',linewidth=1)
    plt.plot(predict_plot,  label='CNN_LSTM_5',color='y',linewidth=1)
    if divider > 0:
        maxVal = max(true_plot.max(),predict_plot.max())
        minVal = min(true_plot.min(),predict_plot.min())

        plt.plot([divider,divider],[minVal,maxVal],label='train/test limit',color='k')


    plt.ylabel('Active power consumed [W]')
    plt.xlabel('Time [/min]')
    plt.legend()
    plt.show()

def plot2(true, predicted, divider):

    predict_plot = scaler.inverse_transform(predicted[0])
    true_plot = scaler.inverse_transform(true[0])

    predict_plot = predict_plot[:,0]
    true_plot = true_plot[:,0]

    plt.figure(figsize=(16,6))
    plt.plot(true_plot, label='True',linewidth=1)
    plt.plot(predict_plot,  label='CNN_LSTM_5',color='y',linewidth=1)

    if divider > 0:
        maxVal = max(true_plot.max(),predict_plot.max())
        minVal = min(true_plot.min(),predict_plot.min())

    plt.ylabel('Active power consumed [W]')
    plt.xlabel('Time [/min]')
    plt.legend()
    plt.show()

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

plot(y_predict_true,y_predict_model,train_size)
plot(y_predict_true[:,-2*test_size:],y_predict_model[:,-2*test_size:],test_size)
plot2(y_test,y_predict_model2,test_size)
