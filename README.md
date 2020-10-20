# EPS-Jetson-Nano
Energy Prediction System with a neural network (CNN-LSTM) in a Jetson Nano. 

In this proyect we'll going to build an active power meter with an Arduino Uno. The data will be send to the Jetson-Nano with the Python script "arduino_serial.py" to stablish the comunication between the Jetson and the Arduino. 
The second script, "neural_training.py" is to start the training for the hybrid neural network and visualize the data. 

# The neural network
CNN-LSTM neural network, which combines convolutional neural network (CNN) and long short-term memory (LSTM), can extract complex features of energy consumption. The CNN layer can extract the features between several variables affecting energy consumption, and the LSTM layer is appropriate for modeling temporal information of irregular trends in time series components. 

The CNN-LSTM method for predicting electric energy consumption consists of a series connection of CNN and LSTM. CNN-LSTM can extract complex features among multiple sensor variables collected for electric energy demand forecasting and can store complex irregular trends. First, the upper layer of CNN-LSTM con-
sists of CNN. The CNN layer can receive various variables that affect electric energy consumption such as voltage, intensity, and sub
metering. In addition, household characteristics such as date, time, behavior of the residents, and household occupancy can also be
modeled as meta information in the CNN layer. CNN consists of an input layer that accepts sensor variables as inputs, an output layer
that extracts features to LSTMs, and several hidden layers.

CNN-LSTM structure:
<img src="images/CNN-LSTM-STRUCT.png">

# Materials:

* 1 - Jetson-Nano Developer Kit from Nvidia
* 1 - HDMI cable
* 1 - LCD screen
* 1 - AC/DC Charger 5v, 4 Amp
* 1 - Arduino uno
* 1 - ZMPT101.B Voltage Sensor
* 1 - SCT-030 Current Transformer Sensor
* 1 - Operational Amplifier
* 1 - 500mA Fuse

# The elecronic circuit for the power meter with the Arduino

* Voltage Sensor:
<img src="images/image.WO57R0.png">

* Current Sensor:
<img src="images/image.RE7AS0.png">

* General Circuit: 
<img src="images/circuit.png">

NOTE: Be carefull when you connect the sensors, the author don't take any responsability for any damage.
# Install the libraries
```
sudo apt-get install numpy pandas pyserial schedule keras
```
# Change permition of the serial port
```
sudo chmode 772 /dev/ttyAMC0/ 
```
# Upload the code
Connect your Arduino to the Jetson Nano and upload the file: "CONSUME_SERIAL.ino"
Remember you need to install the libraries:
* Filters.h
* avr/wdt.h
* Wire.h

# Run the code
```
cd /EPS-Jetson-Nano/ 
python3 arduino_serial.py
```
In other terminal run:
```
cd /EPS-Jetson-Nano/ 
python3 neural_training.py
```
Wait to acquire enough data 

# Final steps:
Use "visualize.py" to visualiaze your predictions of the .h5 file saved after the deep learning training.
