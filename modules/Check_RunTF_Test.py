# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Flatten
from tensorflow.keras.layers import ConvLSTM2D

import numpy as np
import time

# import keras

import tensorflow as tf


# tf.compat.v1.InteractiveSession() #3-4ms
# with tf.compat.v1.Session():
# None

size = int(1e5)
N = 500
W_nodes = 800

X = np.random.random((size, N))
Y = np.random.random(size)

print("\n" * 3)
print("Testing model (no session)")
model = Sequential()
model.add(Dense(W_nodes, input_shape=(N,)))
model.add(Dense(W_nodes))
model.add(Dense(W_nodes))
model.add(Dense(W_nodes))
model.add(Dense(W_nodes))
model.add(Dense(W_nodes))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

print(f"Fitting X:{X.shape}, Y:{Y.shape}")
t0 = time.time()
model.fit(X, Y, verbose=True, epochs=1, batch_size=100)
model.predict(X)
tend = time.time()
print(f"Time fit i predict: {tend - t0}")

####################


print("Checking Compat V1 Session")
import tensorflow as tf


with tf.compat.v1.Session():
    model = Sequential()
    model.add(Dense(W_nodes, input_shape=(N,)))
    model.add(Dense(W_nodes))
    model.add(Dense(W_nodes))
    model.add(Dense(W_nodes))
    model.add(Dense(W_nodes))
    model.add(Dense(W_nodes))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    print(f"Fitting X:{X.shape}, Y:{Y.shape}")
    t0 = time.time()
    model.fit(X, Y, verbose=True, epochs=1, )
    model.predict(X, verbose=True)
    tend = time.time()
    print(f"Time fit i predict: {tend - t0}")
