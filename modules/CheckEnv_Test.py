import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Flatten
from tensorflow.keras.layers import ConvLSTM2D

import numpy as np


# tf.compat.v1.InteractiveSession() #3-4ms
# with tf.compat.v1.Session():
# None

N = int(1e5)
X = np.random.random((N, 20))
Y = np.random.random(N)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# config = tf.compat.v1.ConfigProto(device_count={'GPU': 1}, )
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
# )

# with tf.compat.v1.Session(config=config):
model = Sequential()
model.add(Dense(50, input_shape=(20,)))
# model.add(LSTM(50))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(X, Y, verbose=True, epochs=3)
