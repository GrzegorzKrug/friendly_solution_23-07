import pytest

from preprocess_data import (
    timestr_to_dayfraction, to_sequences_1d, to_sequences_2d,
    to_sequences_forward,
)
import numpy as np


baseparams = [
        ("00:00:00.0", 0),
        ("00:00:00.000", 0),
        ("00:00:00.000000", 0),
        ("24:00:00.0", 1),
        ("24:00:00.000", 1),
        ("24:00:00.000000", 1),


        # Should still be valid
        ("-00:00:00.000000", 0),
        ("00:-00:00.000000", 0),
        ("00:00:-00.000000", 0),
        ("24:-00:00.000000", 1),
        ("24:00:-00.000000", 1),

]

errorparams = [

        # Negation
        ("-01:00:00.000000", 0),
        ("-1:00:00.000000", 0),
        ("00:-10:00.000000", 0),
        ("00:-01:00.000000", 0),
        ("00:00:-1.000000", 0),
        ("00:00:-01.000000", 0),
        ("00:00:00.-1", 0),
        ("00:00:00.-00001", 0),
        ("00:00:00.-000001", 0),

        # To many chars in us
        ("00:00:00.0000000", 0),
        ("24:00:00.0000000", 1),
        ("00:00:00.-100000", 0),
        ("00:00:00.-000000", 0),
        ("24:00:00.-000000", 1),

        # Time limit
        ("25:00:00.0000000", 1),
        ("15:65:00.0000000", 1),
        ("00:00:65.0000000", 1),
]

fraction_params = [
        # Fractions
        ("12:00:00.000000", 0.5),
        ("01:00:00.000000", 1 / 24),
        ("02:0:0.000000", 2 / 24),
        ("03:0:0.000000", 3 / 24),

        # Micro Seconds
        ("00:00:00.100000", 0.1 / 60 / 60 / 24),
        ("00:00:00.010000", 0.01 / 60 / 60 / 24),
        ("00:00:00.001000", 0.001 / 60 / 60 / 24),
        ("00:00:00.000100", 0.0001 / 60 / 60 / 24),
        ("00:00:00.000010", 0.00001 / 60 / 60 / 24),
        ("00:00:00.000001", 0.000001 / 60 / 60 / 24),
]


@pytest.mark.parametrize("inp,cor", baseparams)
def test1_valid_accurate(inp, cor):
    ret = timestr_to_dayfraction(inp)
    assert ret == cor, f"Value should be: {cor}, but it is: {ret}"


@pytest.mark.parametrize("inp,cor", errorparams)
def test2_errors(inp, cor):
    with pytest.raises(ValueError) as e_value:
        timestr_to_dayfraction(inp)


@pytest.mark.parametrize("inp,cor", fraction_params)
def test3_fractions(inp, cor, eps=1e-12):
    ret = timestr_to_dayfraction(inp)
    diff = abs(ret - cor)
    assert diff <= eps, f"Value should be: {cor}, but it is: {ret} and diff: {diff} > {eps}"


def test4_sequences():
    "Scripted test"
    ft_num = 2
    fwd_intervals = [1]
    y_size = len(fwd_intervals)
    seq_size = 3

    arr = np.arange(20).reshape(-1, ft_num)
    print("\nInput:")
    print(arr)
    x, y = to_sequences_forward(arr, seq_size, fwd_intervals=fwd_intervals)
    print("X output:")
    print(x)
    print("Y output:")
    print(y)
    print(x.shape, y.shape)

    assert x[0][0][0] == arr[0, 0]
    assert x[0][1][0] == arr[1, 0]
    assert x[0][2][0] == arr[2, 0]

    assert x[0][0][1] == arr[0, 1]
    assert x[0][1][1] == arr[1, 1]
    assert x[0][2][1] == arr[2, 1]

    assert x.shape[1] == seq_size
    assert y.shape[1] == y_size

    assert y.shape[2] == ft_num
    assert x.shape[2] == ft_num

    # assert x[0][0][0] == arr[0, 0]
    # assert x[0][0][1] == arr[0, 1]
    # assert x[0][1][0] == arr[1, 0]
    # assert x[0][1][1] == arr[1, 1]

    "All Segments"
    assert y.shape[0] == 7

    assert y[0, 0, 0] == arr[3, 0]
    assert y[1, 0, 0] == arr[4, 0]
    assert y[2, 0, 0] == arr[5, 0]

    assert y[0, 0, 1] == arr[3, 1]
    assert y[1, 0, 1] == arr[4, 1]
    assert y[2, 0, 1] == arr[5, 1]
