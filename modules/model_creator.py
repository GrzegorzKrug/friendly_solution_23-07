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

from tensorflow import keras

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

from keras.optimizers import Adam

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
            # print(f"Register f: {func.__name__}, {func} at: {num}")
            if num in cls.funcs:
                raise KeyError(f"Function is registred already at num: {num}, ")
            cls.funcs[num] = func
            return func

        return wrapper


def compile_decorator(**optimkwargs):
    def outer_wrapper(fun):
        # print("kwargs:", optimkwargs)
        opt_kw_params = dict(optimizer='adam', loss='mae')
        opt_kw_params.update(optimkwargs)

        @wraps(fun)
        def wrapper(*a, compile=True, **kw):
            model = fun(*a, **kw)
            if compile:
                print(f"Compiling with params: {opt_kw_params}")
                model.compile(**opt_kw_params, metrics=['mse', 'mae', 'logcosh'])

            return model

        return wrapper

    return outer_wrapper


@ArchRegister.register(1)
@compile_decorator()
def arch_1(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            float_only_nodes=1,
    )

    return model


@ArchRegister.register(2)
@compile_decorator()
def arch_2(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            lst_on_left=2, float_only_nodes=1,
    )

    return model


@ArchRegister.register(3)
@compile_decorator()
def arch_3(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            lst_on_left=2, common_nodes=2, float_only_nodes=2
    )

    return model


@ArchRegister.register(4)
@compile_decorator()
def arch_4(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2pipes(time_feats, time_window, float_feats, out_size, nodes, float_only_nodes=6)

    return model


@ArchRegister.register(5)
@compile_decorator()
def arch_5(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2pipes(time_feats, time_window, float_feats, out_size, nodes, common_nodes=3)

    return model


@ArchRegister.register(6)
@compile_decorator()
def arch_6(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_pyramid(
            float_feats, nodes, out_size, time_feats, time_window,
            pyramid_max=3, pyramid_min=1,
            float_only_nodes=2,
            common_nodes=2, )

    return model


@ArchRegister.register(7)
@compile_decorator()
def arch_7(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_pyramid(
            float_feats, nodes, out_size, time_feats, time_window,
            pyramid_max=4, pyramid_min=2,
            float_only_nodes=2,
            common_nodes=2, )

    return model


@ArchRegister.register(101)
@compile_decorator()
def arch_101(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2_flats(
            time_feats, time_window, float_feats, out_size, nodes,

    )
    return model


@ArchRegister.register(102)
@compile_decorator()
def arch_101(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2_flats(
            time_feats, time_window, float_feats, out_size, nodes,
            dense_on_right=1,
    )
    return model


def builder_2_flats(
        time_feats, time_window, float_feats, out_size, nodes,
        dens_on_left=1, dense_on_right=0, common_nodes=1,
        act_L='relu', act_R='relu',
):
    """
    2 Pipe lines
    Args:
        float_feats:
        nodes:
        out_size:
        time_feats:
        time_window:
        dens_on_left:
        dense_on_right:
        common_nodes:

    Returns:

    """
    time_input = Input(shape=(time_window, time_feats), )
    float_input = Input(shape=(float_feats,), )

    "Time series LSTM"
    # input_L = tf.reshape(time_input, (-1, time_window, time_feats))
    flat_input = Flatten()(time_input)
    if dens_on_left > 1:
        den_L = Dense(nodes, activation=act_L)(flat_input)
        for i in range(dens_on_left - 2):
            den_L = Dense(nodes, activation=act_L)(den_L)
        den_L = Dense(nodes, activation=act_L)(den_L)
    else:
        den_L = Dense(nodes, activation=act_L)(flat_input)
        # print("LSTM")
        # print(den_L.shape)

    "Float section"
    fl_dense = float_input
    if dense_on_right > 0:
        for i in range(dense_on_right):
            fl_dense = Dense(nodes, activation=act_R)(fl_dense)

    conc = Concatenate(axis=1)([den_L, fl_dense])

    "Common section"
    dens = conc
    if common_nodes > 0:
        for i in range(common_nodes):
            dens = Dense(nodes, activation='relu')(dens)

    last = Dense(out_size, activation='linear')(dens)
    "Assign inputs / outputs"
    model = Model(inputs=[time_input, float_input], outputs=last)
    return model


def builder_2pipes(
        time_feats, time_window, float_feats, out_size, nodes,
        lst_on_left=1, float_only_nodes=0, common_nodes=1):
    """
    2 Pipe lines
    Args:
        float_feats:
        nodes:
        out_size:
        time_feats:
        time_window:
        lst_on_left:
        float_only_nodes:
        common_nodes:

    Returns:

    """
    time_input = Input(shape=(time_window, time_feats), )
    float_input = Input(shape=(float_feats,), )

    "Time series LSTM"
    ls_inp = tf.reshape(time_input, (-1, time_window, time_feats))
    if lst_on_left > 1:
        lstm_l = LSTM(nodes, return_sequences=True)(ls_inp)
        for i in range(lst_on_left - 2):
            lstm_l = LSTM(nodes, return_sequences=True)(lstm_l)
        lstm_l = LSTM(nodes)(lstm_l)
    else:
        lstm_l = LSTM(nodes)(ls_inp)
        # print("LSTM")
        # print(lstm_l.shape)

    "Float section"
    fl_dense = float_input
    if float_only_nodes > 0:
        for i in range(float_only_nodes):
            fl_dense = Dense(nodes, activation='relu')(fl_dense)

    ls1 = Flatten()(lstm_l)
    conc = Concatenate(axis=1)([ls1, fl_dense])

    "Common section"
    dens = conc
    if common_nodes > 0:
        for i in range(common_nodes):
            dens = Dense(nodes, activation='relu')(dens)

    last = Dense(out_size, activation='linear')(dens)
    "Assign inputs / outputs"
    model = Model(inputs=[time_input, float_input], outputs=last)
    return model


def builder_pyramid(
        float_feats, nodes, out_size, time_feats, time_window,
        pyramid_max=1, pyramid_min=1, float_only_nodes=0, common_nodes=1,

):
    """
    2
    Args:
        float_feats:
        nodes:
        out_size:
        time_feats:
        time_window:
        pyramid_max:
        float_only_nodes:
        common_nodes:

    Returns:

    """
    time_input = Input(shape=(time_window, time_feats), )
    float_input = Input(shape=(float_feats,), )

    "Time series LSTM"
    ls_inp = tf.reshape(time_input, (-1, time_window, time_feats))
    arr = [Flatten()(make_flat_lstm_sequence(ls_inp, nodes, val))
           for val in range(pyramid_max + 1) if val >= pyramid_min
           ]

    "Float section"
    fl_dense = float_input
    if float_only_nodes > 0:
        for i in range(float_only_nodes):
            fl_dense = Dense(nodes, activation='relu')(fl_dense)

    conc = Concatenate(axis=1)([*arr, fl_dense])

    "Common section"
    dens = conc
    if common_nodes > 0:
        for i in range(common_nodes):
            dens = Dense(nodes, activation='relu')(dens)

    last = Dense(out_size, activation='linear')(dens)
    "Assign inputs / outputs"
    model = Model(inputs=[time_input, float_input], outputs=last)
    return model


def make_flat_lstm_sequence(ls_inp, nodes, how_many):
    if how_many > 1:
        lstm_l = LSTM(nodes, return_sequences=True)(ls_inp)
        for i in range(how_many - 2):
            lstm_l = LSTM(nodes, return_sequences=True)(lstm_l)
        lstm_l = LSTM(nodes)(lstm_l)
    else:
        lstm_l = LSTM(nodes)(ls_inp)
    return lstm_l


models_folder = os.path.join(os.path.dirname(__file__), "..", "models", "")


def grid_models_generator(time_feats, time_window, float_feats, out_size):
    counter = 1
    for arch_num in [101, 2, 102]:
        for nodes in [300]:
            for batch in [2000]:
                for loss in ['huber', 'mae']:
                    for lr in [1e-7, 1e-8]:
                        print(f"Yielding params counter: {counter}")
                        yield counter, (
                                arch_num, time_feats, time_window, float_feats, out_size,
                                nodes, lr, batch, loss
                        )
                        counter += 1


def model_builder(
        arch_num, time_feats, time_window, float_feats, out_size,
        loss, nodes, lr, batch):
    arch = ArchRegister.funcs[arch_num]
    # model = None
    model = arch(
            time_feats, time_window, float_feats, out_size, nodes,
            compile=False
    )
    model: keras.Model
    # model._init_set_name(f"{counter}-{arch_num}")
    # model.name = f"{counter}-{arch_num}"
    print(f"Compiling model with lr: {lr}")
    adam = Adam(learning_rate=lr)
    model.compile(loss=loss, optimizer=adam)
    return model


def plot_all_architectures():
    for name, func in ArchRegister.funcs.items():
        print(name, func)
        model = func(1, 1, 1, 1)
        keras.utils.plot_model(model, models_folder + f"{name}.png")


if __name__ == "__main__":
    plot_all_architectures()
