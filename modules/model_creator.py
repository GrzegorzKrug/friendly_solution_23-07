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
def arch_1(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            float_only_nodes=0,
    )

    return model


@ArchRegister.register(2)
@compile_decorator()
def arch_2(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            lst_on_left=1, float_only_nodes=1,
    )

    return model


@ArchRegister.register(3)
@compile_decorator()
def arch_3(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            lst_on_left=1, common_nodes=2, float_only_nodes=0
    )

    return model


@ArchRegister.register(4)
@compile_decorator()
def arch_4(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            lst_on_left=2, common_nodes=1, float_only_nodes=0
    )

    return model


@ArchRegister.register(5)
@compile_decorator()
def arch_5(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_2pipes(
            time_feats, time_window, float_feats, out_size, nodes,
            lst_on_left=2, common_nodes=2, float_only_nodes=0
    )

    return model


@ArchRegister.register(40)
@compile_decorator()
def arch_40(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_2pipes(time_feats, time_window, float_feats, out_size, nodes, float_only_nodes=6)

    return model


@ArchRegister.register(50)
@compile_decorator()
def arch_55(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_2pipes(time_feats, time_window, float_feats, out_size, nodes, common_nodes=3)

    return model


@ArchRegister.register(60)
@compile_decorator()
def arch_60(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_pyramid(
            float_feats, nodes, out_size, time_feats, time_window,
            pyramid_max=3, pyramid_min=1,
            float_only_nodes=2,
            common_nodes=2, )

    return model


@ArchRegister.register(70)
@compile_decorator()
def arch_70(time_feats, time_window, float_feats, out_size, nodes=20, iteration=0):
    """"""
    model = builder_pyramid(
            float_feats, nodes, out_size, time_feats, time_window,
            pyramid_max=4, pyramid_min=2,
            float_only_nodes=2,
            common_nodes=2, )

    return model


@ArchRegister.register(101)
@compile_decorator()
def arch_101(
        time_feats, time_window, float_feats, out_size, nodes=20,
        reg_k=None, reg_b=None, reg_out_k=None, reg_out_b=None,
        activation='relu',
):
    """"""
    model = builder_2_flats(
            time_feats, time_window, float_feats, out_size, nodes,
            reg_k=reg_k, reg_b=reg_b, reg_out_k=reg_out_k, reg_out_b=reg_out_b,
            act_L=activation, act_R=activation, act_comm=activation,

    )
    return model


@ArchRegister.register(102)
@compile_decorator()
def arch_102(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2_flats(
            time_feats, time_window, float_feats, out_size, nodes,
            dense_on_right=1,
    )
    return model


@ArchRegister.register(103)
@compile_decorator()
def arch_103(
        time_feats, time_window, float_feats, out_size, nodes=20,
        reg_k=None, reg_b=None, reg_out_k=None, reg_out_b=None,
        activation='relu',

):
    """"""
    model = builder_2_flats(
            time_feats, time_window, float_feats, out_size, nodes, common_nodes=2,
            reg_k=reg_k, reg_b=reg_b, reg_out_k=reg_out_k, reg_out_b=reg_out_b,
            act_L=activation, act_R=activation, act_comm=activation,

    )
    return model


@ArchRegister.register(104)
@compile_decorator()
def arch_104(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2_flats(
            time_feats, time_window, float_feats, out_size, nodes, dens_on_left=2, common_nodes=1,
    )
    return model


@ArchRegister.register(105)
@compile_decorator()
def arch_105(time_feats, time_window, float_feats, out_size, nodes=20):
    """"""
    model = builder_2_flats(
            time_feats, time_window, float_feats, out_size, nodes, dens_on_left=2, common_nodes=2,
    )
    return model


def builder_2_flats(
        time_feats, time_window, float_feats, out_size, nodes,
        dens_on_left=1, dense_on_right=0, common_nodes=1,
        act_L='relu', act_R='relu', act_comm='relu', act_out='linear',
        reg_k=None, reg_b=None, reg_out_k=None, reg_out_b=None,
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

    "Time series"
    # input_L = tf.reshape(time_input, (-1, time_window, time_feats))
    flat_input = Flatten()(time_input)
    if dens_on_left > 1:
        den_L = Dense(
                nodes, activation=act_L,
                bias_regularizer=reg_b, kernel_regularizer=reg_k,
        )(flat_input)
        for i in range(dens_on_left - 2):
            den_L = Dense(
                    nodes, activation=act_L,
                    bias_regularizer=reg_b, kernel_regularizer=reg_k,
            )(den_L)
        den_L = Dense(
                nodes, activation=act_L,
                bias_regularizer=reg_b, kernel_regularizer=reg_k,
        )(den_L)
    else:
        den_L = Dense(
                nodes, activation=act_L,
                bias_regularizer=reg_b, kernel_regularizer=reg_k,
        )(flat_input)
        # print("LSTM")
        # print(den_L.shape)

    "Float section"
    fl_dense = float_input
    if dense_on_right > 0:
        for i in range(dense_on_right):
            fl_dense = Dense(
                    nodes, activation=act_R,
                    bias_regularizer=reg_b, kernel_regularizer=reg_k,
            )(fl_dense)

    conc = Concatenate(axis=1)([den_L, fl_dense])

    "Common section"
    dens = conc
    if common_nodes > 0:
        for i in range(common_nodes):
            dens = Dense(
                    nodes, activation=act_comm,
                    bias_regularizer=reg_b, kernel_regularizer=reg_k,
            )(dens)

    last = Dense(out_size, activation=act_out,
                 bias_regularizer=reg_out_b, kernel_regularizer=reg_out_k,
                 )(dens)
    "Assign inputs / outputs"
    model = Model(inputs=[time_input, float_input], outputs=last)
    return model


def builder_2pipes(
        time_feats, time_window, float_feats, out_size, nodes,
        lst_on_left=1, float_only_nodes=0, common_nodes=1):
    """
    2 Pipe lines with LSTM
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
    counter = 0
    for batch in [500]:
        for arch_num in [1, 3, 101, 103]:
            for dc in [0.9]:
                for nodes in [1000]:
                    for loss in ['huber', 'mae', ]:
                        for lr in [1e-5, 1e-6]:
                            # for arch_num in [103]:
                            #     for dc in [0.9, 0]:
                            #         for nodes in [300]:
                            #             for batch in [1000, 2000]:
                            #                 for loss in ['huber', 'mae', 'mse']:
                            #                     for lr in [1e-5, 1e-6, 1e-7]:
                            print(f"Yielding model, counter: {counter}")
                            iteration = 0
                            yield counter, (
                                    arch_num, time_feats, time_window, float_feats, out_size,
                                    nodes, lr, batch, loss, dc, iteration
                            )
                            counter += 1


def grid_models_generator_2(time_feats, time_window, float_feats, out_size):
    counter = 0
    for batch in [300, 500]:
        for arch_num in [101, 103]:
            for dc in [0.9]:
                for nodes in [1000]:
                    for loss in ['huber', 'mae', ]:
                        for lr in [1e-5, 1e-6]:
                            # for arch_num in [103]:
                            #     for dc in [0.9, 0]:
                            #         for nodes in [300]:
                            #             for batch in [1000, 2000]:
                            #                 for loss in ['huber', 'mae', 'mse']:
                            #                     for lr in [1e-5, 1e-6, 1e-7]:
                            print(f"Yielding model, counter: {counter}")
                            iteration = 0
                            yield counter, (
                                    arch_num, time_feats, time_window, float_feats, out_size,
                                    nodes, lr, batch, loss, dc, iteration
                            )
                            counter += 1


def grid_models_generator_it23(time_feats, time_window, float_feats, out_size):
    counter = 0
    for batch in [500]:
        for lr in [1e-6]:
            for arch_num in [101, 103]:
                for dc in [0.9]:
                    for iteration in [0, 16, 17, 18]:
                        for nodes in [3000, ]:
                            for loss in ['huber']:
                                print(f"Yielding model, counter: {counter}")
                                yield counter, (
                                        arch_num, time_feats, time_window, float_feats, out_size,
                                        nodes, lr, batch, loss, dc, iteration
                                )
                                counter += 1


def model_builder(
        arch_num, time_feats, time_window, float_feats, out_size,
        loss, nodes, lr, iteration=0, override_params=dict(), ):
    arch = ArchRegister.funcs[arch_num]
    # print(f"Compiling model: "
    #       f"{arch_num}({iteration})-{time_feats}x{time_window}&{float_feats} -> {out_size}, L:{loss} No:{nodes}, Lr:{lr}")

    arch_num = override_params.get("arch_num", arch_num)
    time_feats = override_params.get("time_feats", time_feats)
    time_window = override_params.get("time_window", time_window)
    float_feats = override_params.get("float_feats", float_feats)
    out_size = override_params.get('out_size', out_size)
    loss = override_params.get("loss", loss)
    nodes = override_params.get('nodes', nodes)
    lr = override_params.get("lr", lr)
    iteration = override_params.get("iteration", iteration)

    "DEFAULT PARAMETERS"
    reg_k = None
    reg_b = None
    reg_out_k = None
    reg_out_b = None
    activation = 'relu'

    if iteration in [0, 1] or iteration is None:
        pass

    elif iteration in [2]:
        reg_k = keras.regularizers.L1(0.001)
        reg_b = keras.regularizers.L1(0.001)
    elif iteration in [3]:
        reg_k = keras.regularizers.L1(0.0001)
        reg_b = keras.regularizers.L1(0.0001)
    elif iteration in [4]:
        reg_k = keras.regularizers.L1(0.00001)
        reg_b = keras.regularizers.L1(0.00001)
    elif iteration in [5]:
        reg_out_b = keras.regularizers.L1(1e-6)
        reg_out_k = keras.regularizers.L1(1e-5)
    elif iteration in [6, 16]:
        reg_k = keras.regularizers.L1(1e-6)
        reg_b = keras.regularizers.L1(1e-6)
        reg_out_b = keras.regularizers.L1(1e-6)
        reg_out_k = keras.regularizers.L1(1e-6)
    elif iteration in [7, 17]:
        reg_k = keras.regularizers.L2(1e-6)
        reg_b = keras.regularizers.L2(1e-6)
        reg_out_b = keras.regularizers.L2(1e-6)
        reg_out_k = keras.regularizers.L2(1e-6)
    elif iteration in [8, 18]:
        reg_k = keras.regularizers.L2(1e-6)
        reg_b = keras.regularizers.L2(1e-5)
    else:
        raise ValueError(f"Iteration not implemented: {iteration}")

    if iteration in [16, 17, 18]:
        activation = 'softsign'

    model = arch(
            time_feats, time_window, float_feats, out_size, nodes,
            reg_k=reg_k, reg_b=reg_b, reg_out_k=reg_out_k, reg_out_b=reg_out_b,
            compile=False,
    )
    print(f"Compiling model: "
          f"{arch_num}({iteration})(@{activation}) - {time_feats}x{time_window}, {float_feats} -> {out_size}, L:{loss} No:{nodes}, Lr:{lr}")
    model: keras.Model
    # model._init_set_name(f"{counter}-{arch_num}")
    # model.name = f"{counter}-{arch_num}"
    adam = Adam(learning_rate=lr, clipnorm=2.0, clipvalue=10)
    model.compile(loss=loss, optimizer=adam)
    return model


def plot_all_architectures():
    for name, func in ArchRegister.funcs.items():
        print(name, func)
        model = func(1, 1, 1, 1)
        keras.utils.plot_model(model, models_folder + f"{name}.png")
        del model
        tf.keras.backend.clear_session()


def show_all_activations():
    # Define input values
    x = np.linspace(-10, 10, 400)

    # Compute activation function values
    relu_values = tf.keras.activations.relu(x)
    leaky_relu_layer = tf.keras.layers.LeakyReLU(alpha=0.2)
    leaky_relu_values = leaky_relu_layer(x)
    sigmoid_values = tf.keras.activations.sigmoid(x)
    tanh_values = tf.keras.activations.tanh(x)
    elu_values = tf.keras.activations.elu(x)
    swish_values = tf.keras.activations.swish(x)
    softplus_values = tf.keras.activations.softplus(x)
    softsign_values = tf.keras.activations.softsign(x)
    gelu_values = tf.keras.activations.gelu(x)

    # Create subplots
    plt.figure(figsize=(12, 8))

    activation_functions = [
            ('ReLU', relu_values),
            ('Leaky ReLU', leaky_relu_values),
            ('Sigmoid', sigmoid_values),
            ('Tanh', tanh_values),
            ('ELU', elu_values),
            ('Swish', swish_values),
            ('Softplus', softplus_values),
            ('Softsign', softsign_values),
            ('GELU', gelu_values),
    ]

    rows = 3
    cols = 3

    for idx, (name, values) in enumerate(activation_functions, start=1):
        plt.subplot(rows, cols, idx)
        plt.plot(x, values)
        plt.title(name)
        plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_all_architectures()
    # model_builder(101, 17, 30, 1, 3, 'mae', 30, 1e-3, 1, override_params=dict(lr=1e-5))
    # model_builder(103, 17, 30, 1, 3, 'mae', 30, 1e-3, 1, override_params=dict(lr=1e-5))
    show_all_activations()
