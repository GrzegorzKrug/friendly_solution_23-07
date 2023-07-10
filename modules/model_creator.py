import time
import os
import sys
import argparse

from collections import deque
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    # Conv1D, Conv2D, MaxPool2D,
    # AveragePooling2D,
    Flatten, concatenate, Concatenate,
    LSTM,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model

import keras

from common_functions import NamingClass

from functools import wraps


# from reward_function import get_reward, get_reward_without_boost
# from reward_function import RewardStore


class ArchRegister:
    funcs = {}

    @classmethod
    def register(cls, num):
        # @wraps(func)
        def wrapper(func):
            print(f"Register f: {func.__name__}, {func} at: {num}")
            if num in cls.funcs:
                raise KeyError(f"Function is registred already at num: {num}, ")
            cls.funcs[num] = func
            return func

        return wrapper


def compile_decorator(fun):
    @wraps(fun)
    def wrapper(*a, compile=True, **kw):
        model = fun(*a, **kw)
        if compile:
            model.compile(optimizer='adam', loss='mae')
        return model

    return wrapper


@ArchRegister.register(1)
@compile_decorator
def arch_1(tser_size, tser_outsize, ft_size=0):
    model = Sequential()
    model.add(LSTM(600, return_sequences=False, input_shape=(1, tser_size)))
    model.add(Dense(50))
    model.add(Dense(tser_outsize, activation='linear'))
    model.compile(optimizer='adam', loss='mae')

    return model


@ArchRegister.register(2)
@compile_decorator
def arch_2(tser_size=5, tser_outsize=1, tser_features=1, ft_size=1, nodes=100):
    # model = Sequential()
    # model = Model(inputs=Input(shape=(20, 1)), outputs=(5,))
    # model = model(Input(shape=(20, 1)))
    layer_in = Input(shape=(tser_size + ft_size,))
    print(f"Inp: {layer_in.shape}")

    layes = tf.split(layer_in, [tser_size, ft_size], axis=1)
    # print(f"Lays0: {layes[0].shape}")
    # print(f"Lays1: {layes[1].shape}")

    ls_inp = tf.reshape(layes[0], (-1, tser_size, tser_features))
    # print("Reshape:", ls_inp.shape)
    ls1 = LSTM(nodes)(ls_inp)
    ls1 = Flatten()(ls1)

    ls2 = LSTM(nodes, return_sequences=True)(ls_inp)
    ls2 = LSTM(nodes)(ls2)
    ls2 = Flatten()(ls2)

    conc = Concatenate(axis=1)([ls1, ls2, layes[1]])

    dens = Dense(nodes, activation='relu')(conc)
    last = Dense(tser_outsize, activation='linear')(dens)
    model = Model(inputs=layer_in, outputs=last)

    return model


@ArchRegister.register(3)
@compile_decorator
def arch_3(time_window=5, tser_outsize=1, time_f=1, ft_size=1, nodes=100):
    # model = Sequential()
    # model = Model(inputs=Input(shape=(20, 1)), outputs=(5,))
    # model = model(Input(shape=(20, 1)))
    layer_in = Input(shape=(time_window + ft_size,))
    print(f"Inp: {layer_in.shape}")

    layes = tf.split(layer_in, [time_window, ft_size], axis=1)
    # print(f"Lays0: {layes[0].shape}")
    # print(f"Lays1: {layes[1].shape}")

    ls_inp = tf.reshape(layes[0], (-1, time_window, time_f))
    # print("Reshape:", ls_inp.shape)
    ls1 = LSTM(nodes)(ls_inp)
    ls1 = Flatten()(ls1)

    ls2 = LSTM(nodes, return_sequences=True)(ls_inp)
    ls2 = LSTM(nodes)(ls2)
    ls2 = Flatten()(ls2)

    conc = Concatenate(axis=1)([ls1, ls2, layes[1]])

    dens = Dense(nodes, activation='relu')(conc)
    last = Dense(tser_outsize, activation='linear')(dens)
    model = Model(inputs=layer_in, outputs=last)

    return model


data_folder = os.path.join(os.path.dirname(__file__), "..", "models", "")

if __name__ == "__main__":
    naming = NamingClass(2, "", 1, 5, 1, 3, 1, postfix="")

    model = arch_3()

    model: keras.Model
    model_dir = data_folder + naming.path + os.path.sep
    print(f"Making folder: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    fp_path = model_dir + "weights.keras"
    print(f"Saving weights to: {fp_path}")
    model.save_weights(fp_path)
    keras.utils.plot_model(model, to_file=model_dir + "mode.png")
