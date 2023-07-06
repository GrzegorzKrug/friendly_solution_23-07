# import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# # from tensorflow.keras.layers import LSTM, Flatten
# # from tensorflow.keras.layers import ConvLSTM2D

import numpy as np

import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, ConvLSTM2D

import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# tf.compat.v1.InteractiveSession() #3-4ms
# with tf.compat.v1.Session():
# None

N = int(3e4)
X = np.random.random((N, 20))
Y = np.random.random(N)

####################
model = Sequential()
model.add(Dense(50, input_shape=(20,)))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(X, Y, verbose=True, epochs=1)
model.predict(X)

####################

# with tf.compat.v1.Session():
#     model = Sequential()
#     model.add(Dense(50, input_shape=(20,)))
#     model.add(Dense(60))
#     model.add(Dense(60))
#     model.add(Dense(60))
#     model.add(Dense(60))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#
#     model.fit(X, Y, verbose=True, epochs=1)
#
#     model.predict(X)
