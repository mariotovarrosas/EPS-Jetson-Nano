import serial
import time
import schedule
import csv
import numpy as np
import datetime
from datetime import date
import pandas as pd
import os
from os import path

def main_func():
    now = datetime.datetime.now()
    today = now.year
    week = now.isocalendar()[1]
    data=[]
    df=[]

    arduino = serial.Serial('/dev/ttyACM0', 9600)
    print('Establishing serial connection with Arduino...')
    arduino_data = arduino.readline()

    decoded_values = str(arduino_data[0:len(arduino_data)].decode("utf-8"))
    list_values = decoded_values.split('x')

    for item in list_values:
        list_in_floats.append(float(item))

    print(f'Collected readings from Arduino: {list_in_floats[0]}[W] {list_in_floats[1]}[V_ac] {list_in_floats[2]}[I_ac] ')
    time_stamp=str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f'Time stamp: {time_stamp}')
    if path.exists('data_'+str(today)+'W'+str(week)+'.csv') == True:
        data.append([time_stamp, list_in_floats[0],list_in_floats[1],list_in_floats[2]])
        node = pd.DataFrame(data, columns=['Time stamp','Active power consumed [W]','Voltage AC [V_ac]','Current AC [I_ac]'])
        node.to_csv('data_'+str(today)+'W'+str(week)+'.csv', index = None,  mode='a', header=False)
    else:
        data.append([time_stamp, list_in_floats[0],list_in_floats[1],list_in_floats[2]])
        node = pd.DataFrame(data, columns=['Time stamp','Active power consumed [W]','Voltage AC [V_ac]','Current AC [I_ac]'])
        node.to_csv('data_'+str(today)+'W'+str(week)+'.csv', index = None)

    print('Data saved in csv file')
    arduino_data = 0
    list_in_floats.clear()
    list_values.clear()
    data.clear()
    arduino.close()
    print('Connection closed')
    print('<--------------------------------------------------------------------------->')


# ----------------------------------------Main Code------------------------------------
# Declare variables to be used
list_values = []
list_in_floats = []

print('Program started')

# Setting up the Arduino
schedule.every(50).seconds.do(main_func)

while True:
    schedule.run_pending()
    time.sleep(4)
