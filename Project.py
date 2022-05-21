# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:03:13 2022

@author: pavan
"""


import pandas as pd           #import pandas
import matplotlib.pyplot as plt   #omport matplot library
import numpy as np     #import numpy
import math           #import mathematics library
from sklearn.preprocessing import MinMaxScaler    #import the minmax scaler from the scikit learn
from sklearn.metrics import mean_squared_error    #import the mean squared error function
from keras.models import Sequential      #import sequential pattern
from keras.layers import Dense, Activation  #import the activation functions
from keras.layers import LSTM     #import the LSTM layer
from keras.layers import Dropout  #import the droput function
from sklearn import preprocessing #import different data preprocessing techniques
import datetime           #import datetime
import seaborn as sns     #import seaborn library

data = pd.read_csv("C:\\Users\\pavan\\Downloads\\AAPL (1).csv")  #load the data using the pandas
data['Date']=pd.to_datetime(data['Date'])  #convert the date column into the standard Date datatype format

#EDA on the Data
#data=data.set_index('Date')     #set the index of the dataframe as date column
#df1=data.groupby(['Date'])['Close'].mean()   #group the data based on Date
#plot(df1)    #plot the grouped data

#corr=data.corr()    #calculate the correlation of the data
#ns.heatmap(corr,annot=True)    #plot the heatmap representing the correlation of the data

#plt.figure(figsize = (18,9))    #set the plot size
#plt.plot(range(data.shape[0]),(data['Low']+data['High'])/2.0) #plt the average price of the stock daily
#plt.xticks(range(0,data.shape[0],500),data['Date'].loc[::500]) #plot the specific X axix labels
#plt.xlabel('Date',fontsize=18)  #add X axis label to the plot
#plt.ylabel('Mid Price',fontsize=18)   #add the Y axus label to the data
#plt.show()   #output the plot


df=data[['Close','Open','High','Low','Volume']]   #create a new dataframe with the predictor variables

train=df.iloc[:1177,:]    #create a training set of data
test=df.iloc[1177:1259,:]  #create a testing set of data

scaler = MinMaxScaler(feature_range = (0, 1))  #set the limit for the minmax scaler
trainscaled=scaler.fit_transform(train)   #transfrom the train data using the minmax scaler
testscaled=scaler.fit_transform(test)     #transform the test data using the minmax scaler

X_train = []    #implement a empty list for predictor variables in the training data
y_train = []    #implement a empty llist for outcome variable in the training data
for i in range(60, 1177):     #intialize a for loop
    X_train.append(trainscaled[i-60:i, 1:5])    #append the values into the empty list of predictors
    y_train.append(trainscaled[i, 0])     #append the values into the empty list of prediction variable.
X_train, y_train = np.array(X_train), np.array(y_train)  #create an array of the list 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4)) 
                                                          #reshape the array into the 3d format

#creating a recurrent neural network with LSTM layer
mod = Sequential()
mod.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))  
                      # Adding the first LSTM layer input layer and some Dropout regularisation
mod.add(Dropout(0.2))   
mod.add(LSTM(units = 50, return_sequences = True))   
                            # Adding a second LSTM layer and some Dropout regularisation
mod.add(Dropout(0.2))       #adding the dropout to the layer
mod.add(LSTM(units = 50, return_sequences = True))   
                           # Adding a third LSTM layer and some Dropout regularisation
mod.add(Dropout(0.2))      #adding the dropout to the layer
mod.add(LSTM(units = 50))  # Adding a fourth LSTM layer and some Dropout regularisation
mod.add(Dropout(0.2))      #adding the dropout to the layer
mod.add(Dense(units = 1))    # Adding the output layer

mod.compile(optimizer = 'adam', loss = 'mean_squared_error')     # Compiling the RNN
mod.fit(X_train, y_train, epochs = 100, batch_size = 32)    #train the neural network on the training data

X_test = []    #implement a empty for list for predictors in the testing data
y_test = []    #implement a empty for list for prediction variable in the testing data
for i in range(60,82):      #initialize a for loop
    X_test.append(testscaled[i-60:i, 1:5])     #append the predictors in the testing data set
    y_test.append(testscaled[i, 0])            #append the prediction output from the testing data set
X_test, y_test = np.array(X_test), np.array(y_test)   #create an array from the list 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4)) 
                                              #reshape the array based on the input format

predicted_stock_price = mod.predict(X_test)   #predict the stock price using the neural network

plt.plot(y_test, color = 'red', label = 'Real AAPL Stock Price')   #plot the true stock value of the data
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted AAPL Stock Price')  
                                                            #plot the predicted valuen of the data
plt.title('AAPL Stock Price Prediction')    #add title to the plot
plt.xlabel('Time')     #add X axis label to the plot
plt.ylabel('AAPL Stock Price')  #add Y axis label to the plot
plt.legend()    #add legend to the plot
plt.show()      #output the plot
