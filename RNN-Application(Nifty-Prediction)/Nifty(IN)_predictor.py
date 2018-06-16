# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 09:18:17 2018

@author: abis
"""



import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Nifty(IN)_Training_Data.csv')
##getting only open price for training. [:, 1:2] to getting matrix


#elemenating comma(,) and converting to float
for i in range(0, len(dataset_train.index)):
    temp=dataset_train['Price'].iloc[i]
    flag=0
    for j in range(0, len(temp)):
        if temp[j]==',' :
            dataset_train['Price'].iloc[i]=float(temp[:j]+temp[(j+1):])
            flag=1
    if flag==0 :
        dataset_train['Price'].iloc[i]=float(temp)
print(0)    
training_set = dataset_train.iloc[:, 1:2].values
# Feature Scaling 
## we can apply standardisation or normalisation. here using normalization.
from sklearn.preprocessing import MinMaxScaler
## getting scalling object
sc = MinMaxScaler(feature_range = (0, 1))
##fitting
training_set_scaled = sc.fit_transform(training_set)



X_train = training_set_scaled[0:(len(dataset_train.index)-7)]
y_train = training_set_scaled[7:len(dataset_train.index)]
# Reshaping the input
X_train = np.reshape(X_train, ((len(dataset_train.index)-7), 1, 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
## As here continues outcome is predicted so it s regression.and it is sequence of layer
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 4, activation='sigmoid',  input_shape = (None, 1)))
regressor.add(Dropout(0.2))

'''# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 32,  activation='sigmoid', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 32,  activation='sigmoid', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 32))
regressor.add(Dropout(0.2))'''

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 500, batch_size = 32)

# Part 3 - Making the predictions and visualising the resu
dataset_test = pd.read_csv('Nifty(IN)_Training_Data.csv')

#elemenating comma(,) and converting to float
for i in range(0, len(dataset_test.index)):
    temp=dataset_test['Price'].iloc[i]
    flag=0
    for j in range(0, len(temp)):
        if temp[j]==',' :
            dataset_test['Price'].iloc[i]=float(temp[:j]+temp[(j+1):])
            flag=1
    if flag==0 :
        dataset_test['Price'].iloc[i]=float(temp)
print(0) 

real_nifty_price = dataset_test.iloc[:, 1:2].values


#getting predicted Nifty

inputs=real_nifty_price
inputs=sc.transform(inputs)

print(len(inputs))
inputs=np.reshape(inputs, (len(inputs), 1, 1))

predicted_nifty_price = regressor.predict(inputs)
predicted_nifty_price = sc.inverse_transform(predicted_nifty_price)

# Visualising the results
plt.plot(real_nifty_price, color = 'red', label = 'Real Nifty Price')
plt.plot(predicted_nifty_price, color = 'blue', label = 'Predicted Nifty Price')
plt.title('Nifty Prediction')
plt.xlabel('Time')
plt.ylabel('Nifty Price')
plt.legend()
plt.show()
