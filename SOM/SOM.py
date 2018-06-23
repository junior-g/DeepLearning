# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 05:30:40 2018

@author: abis
"""

# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling(normalizing all values between 0-1)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom  #for implmenting SOM
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) #x X y size grid with sigma(radius), learning rate(amout by witch wegiht updated) 
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
fraud=mappings[(1, 7)]
fraud=sc.inverse_transform(fraud)
fraud1=mappings[(1, 8)]
fraud1=sc.inverse_transform(fraud1)
fraud3=np.concatenate((fraud1, fraud), axis=0)
fraud3 = sc.inverse_transform(fraud3)
frauds = np.concatenate((mappings[(1,0)], mappings[(1,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)