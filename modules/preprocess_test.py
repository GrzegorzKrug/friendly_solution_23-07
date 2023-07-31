import pytest

from preprocess_data import (
    timestr_to_dayfraction, to_sequences_1d, to_sequences_2d, split_df_to_segments,
)
from modules.common_functions import to_sequences_forward

import numpy as np
import pandas as pd


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


arr_list = [
        (np.random.random((20, 5)), 1, 1),
        (np.random.random((20, 5)), 1, 2),
        (np.random.random((20, 5)), 1, 3),
        (np.random.random((20, 5)), 1, 5),

        (np.random.random((20, 5)), 2, 5),
        (np.random.random((20, 5)), 4, 5),
        (np.random.random((20, 5)), 10, 5),
]


@pytest.mark.parametrize("arr,ft_num,seq_size", arr_list)
def test5_sequences(arr, ft_num, seq_size):
    fwd_intervals = [1]
    predict_tser_size = len(fwd_intervals)
    # seq_size = 3
    # ft_num = 2
    # in_size = 50

    # arr = np.arange(50).reshape(-1, ft_num)
    arr = arr.reshape(-1, ft_num)
    h, w = arr.shape
    # print("\nInput:")
    # print(arr)
    x, y = to_sequences_forward(arr, seq_size, fwd_intervals=fwd_intervals)

    print(f"Ft num: {ft_num}, seq: {seq_size}")
    print(x.shape, y.shape)

    "Shape 0"
    assert len(x) == (h + -seq_size), "Wronge size"
    assert len(y) == (h + -seq_size), "Wronge size"

    "Shape 1"
    assert x.shape[1] == seq_size, "Wronge size"
    assert y.shape[1] == predict_tser_size, "Wronge size"

    "Shape 2"
    assert x.shape[2] == ft_num, "Wronge size"
    assert y.shape[2] == ft_num, "Wronge size"


arr_list2 = [
        (np.random.random((10, 2)), 1, 1),
        (np.random.random((10, 2)), 1, 2),
        (np.random.random((10, 2)), 2, 1),
        (np.random.random((10, 2)), 2, 2),
]


@pytest.mark.parametrize("arr,ft_num,seq_size", arr_list2)
def test6_sequences(arr, ft_num, seq_size):
    """Checking values"""
    fwd_intervals = [1]
    arr = arr.reshape(-1, ft_num)

    x, y = to_sequences_forward(arr, seq_size, fwd_intervals=fwd_intervals)

    print(f"Ft num: {ft_num}, seq: {seq_size}")
    print(arr)
    print(arr.shape)
    print(x.shape, y.shape)

    print("===========")
    # print(x)
    print("===========")
    # print(y)

    B, T, F = x.shape
    "Smaller then X due rest values are in Y"
    for b in range(B):
        for t in range(T):
            for f in range(F):
                cur_x = x[b, t, f]
                # print(f"CheckingX: {b + t, f}, b{b}, t{t}, f{f}, {cur_x}")
                check_x = arr[b + t, f]
                assert cur_x == check_x, "Values X missmatch"

    B, T, F = y.shape
    "Smaller then X due rest values are in Y"
    for b in range(B):
        for t in range(T):
            for f in range(F):
                cur_y = y[b, t, f]
                # print(f"CheckingY: {b + t, f}, b{b}, t{t}, f{f}, {cur_y}")
                check_y = arr[b + t + seq_size, f]
                assert cur_y == check_y, "Values Y missmatch"


arr_list3 = [
        (np.random.random((10, 1)), 1, 1),
        (np.random.random((10, 1)), 2, 1),
        (np.random.random((10, 1)), 3, 1),
        (np.random.random((10, 1)), 5, 1),
        (np.random.random((10, 1)), 7, 1),

        (np.random.random((10, 1)), 3, 3),
        (np.random.random((10, 1)), 4, 4),
        (np.random.random((10, 1)), 5, 5),
]


@pytest.mark.parametrize("arr,intv,seq_size", arr_list3)
def test7_intervals(arr, intv, seq_size):
    """Checking values"""
    fwd_intervals = [intv]
    arr = arr.reshape(-1, 1)

    x, y = to_sequences_forward(arr, seq_size, fwd_intervals=fwd_intervals)

    print(f"Intervals: {fwd_intervals}, seq: {seq_size}")
    print(arr)
    print(arr.shape)
    print(x.shape, y.shape)

    assert len(x) == len(arr) - seq_size - max(fwd_intervals) + 1
    assert len(y) == len(arr) - seq_size - max(fwd_intervals) + 1

    B, T, F = y.shape

    for b in range(B):
        for t in range(T):
            for f in range(F):
                fw_y = y[b, t, f]

                # check_y = arr
                check_y = arr[b + t + seq_size + intv - 1, f]
                print(f"Checking: b{b}, t{t}, f{f}, {b + t + seq_size + intv - 1, f}, {check_y}")
                assert fw_y == check_y, "Values Y missmatch"

    # raise ValueError


arr_list3 = [
        (np.random.random((20, 1)), 1, 1),
        (np.random.random((20, 1)), 2, 1),
        (np.random.random((20, 1)), 3, 1),
        (np.random.random((20, 1)), 5, 1),
        (np.random.random((20, 1)), 7, 1),

        (np.random.random((20, 1)), 3, 3),
        (np.random.random((20, 1)), 4, 4),
        (np.random.random((20, 1)), 5, 5),
]


@pytest.mark.parametrize("arr,intv,seq_size", arr_list3)
def test8_intervals_2d(arr, intv, seq_size):
    """Checking values"""
    fwd_intervals = [intv]
    arr = arr.reshape(-1, 2)

    x, y = to_sequences_forward(arr, seq_size, fwd_intervals=fwd_intervals)

    print(f"Intervals: {fwd_intervals}, seq: {seq_size}")
    print(arr)
    print(arr.shape)
    print(x.shape, y.shape)

    assert len(x) == len(arr) - seq_size - max(fwd_intervals) + 1
    assert len(y) == len(arr) - seq_size - max(fwd_intervals) + 1

    B, T, F = y.shape

    for b in range(B):
        for t in range(T):
            for f in range(F):
                fw_y = y[b, t, f]

                # check_y = arr
                check_y = arr[b + t + seq_size + intv - 1, f]
                print(f"Checking: b{b}, t{t}, f{f}, {b + t + seq_size + intv - 1, f}, {check_y}")
                assert fw_y == check_y, "Values Y missmatch"

    raise ValueError


# test_9_args=[
#         (arr),
# ]
@pytest.fixture(params=[5, 10, 25, 50, 100, 200])
def timeseries_df(request):
    N = request.param
    print(f"Creating df of size: {N}")
    df = pd.DataFrame(columns=['timestamp_ns', 'value'], dtype=float)
    arr = np.random.random((N, 2)) * (1e9, 1)
    # arr = np.sort(arr,axis=0)
    # print(arr.shape)
    # df.loc[:N] = arr
    for i, row in enumerate(arr):
        df.loc[i] = row

    return df


@pytest.mark.parametrize("split_gap", (0.2, 0.5,))
def test9_split_test(timeseries_df, split_gap):

    segments = split_df_to_segments(timeseries_df, split_gap)

    assert len(segments) > 0, "Must return some segments"

    assert sum(map(len, segments)) == len(timeseries_df), "Size must match!"

    print(f"Got segments: {len(segments)}")

    for segm in segments:
        if len(segm) > 1:
            diff = np.diff(segm['timestamp_ns'] / 1e9)
            print("Segment:")
            print(segm)
            print("Segment diff:")
            print(diff)
            assert np.sum(diff > split_gap) <= 0, f"Must have smaller gaps only in segment but got: {diff}"
