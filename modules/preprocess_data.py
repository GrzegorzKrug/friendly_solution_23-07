import os
from datetime import timedelta

import multiprocessing as mpc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.style import use
from yasiu_native.time import measure_real_time_decorator

from common_settings import path_data_clean_folder


folder_danych = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dane", "")) + os.path.sep


def clean_col(name):
    return str(name).lower().strip()


def strip_timestr(x):
    tm_rest, usecond = x.split('.')
    hour, minute, second = tm_rest.split(":")
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    if len(usecond) > 6:
        raise ValueError("Too many digits in micro second")
    usecond = int(f"{usecond:<06}")

    if hour < 0 or minute < 0 or second < 0 or usecond < 0:
        raise ValueError(f"Some value is negative: {hour}:{minute}:{second}.{usecond}")

    return hour, minute, second, usecond


def timestr_to_dayfraction(x: str):
    # print(type(x), f"'{x}'")
    hour, minute, second, usecond = strip_timestr(x)

    hour = hour / 24
    minute = minute / 60 / 24
    second = second / 60 / 60 / 24
    usecond = usecond / 1000_000 / 60 / 60 / 24
    daytime = hour + minute + second + usecond

    return daytime


def timestr_to_timedelta(x: str):
    hour, minute, second, usecond = strip_timestr(x)
    # delta = np.timedelta64(f"{hour}:{minute}:{second}")
    delta = timedelta(hours=hour, minutes=minute, seconds=second, microseconds=usecond)
    return delta


def timestr_to_seconds(x: str):
    # print(type(x), f"'{x}'")
    hour, minute, second, usecond = strip_timestr(x)

    seconds = hour * 60 * 60 + minute * 60 + second + usecond / 1e6

    return seconds


def preprocess(df, allow_plot=False, save_path=None, first_sample_date: str = None):
    """

    Args:
        df:
        allow_plot:
        save_path:
        first_sample_date: 'YYYY-MM-DD'
            Do not 0 pad!
            "2023-6-15"

    Returns:

    """
    df = df.copy()
    # print("Processing data")

    if first_sample_date:
        first_sample_date = str(first_sample_date)
        mask = df['Date'] >= first_sample_date
        first_ind = np.argmax(mask)
        print(f"Cutting data to day: {first_ind}: {first_sample_date}")
        df = df.loc[mask, :]
        if len(df) <= 0:
            return df

    df.reset_index(inplace=True)

    "Rename columns"
    df.columns = list(map(clean_col, df.columns))
    # out_df = df.copy()

    "Day fraction"
    df = df.assign(dayfraction=df['time'].apply(timestr_to_dayfraction))
    df = df.assign(seconds=df['time'].apply(timestr_to_seconds))

    time_fmt = "%Y-%m-%d"
    dt_64 = pd.to_datetime(df['date'], format=time_fmt)
    # print(dt_64, dt_64.dtype)
    # print(dt_64.dt.dayofweek)

    tdelta = df['time'].apply(timestr_to_timedelta)
    timestamp = dt_64.values + tdelta
    # print(timestamp.dtype)
    df = df.assign(timestamp_str=timestamp)
    df = df.assign(timestamp_ns=timestamp.astype(np.int64))
    # print(timestamp.astype(int))

    # print(dir(dt_64))
    # print(timestamp)
    # ""
    # df = df.assign(timestamp=timestamp)

    "Day of week"
    df = df.assign(dayofweek=dt_64.dt.dayofweek)

    "Days in month"
    df = df.assign(daysinmonth=dt_64.dt.days_in_month)

    "Week number"
    df = df.assign(weekno=dt_64.dt.isocalendar().week)

    "Month fraction"
    df = df.assign(monthfraction=dt_64.dt.day / dt_64.dt.days_in_month)

    # dt = pd.DataFrame(dt_64)
    # print(dt)
    # print(dt['date'].dayofweek)

    if allow_plot:
        "No plot"
        # plt_df = df.iloc[:1000]
        plt_df = df
        y1 = plt_df['open'].to_numpy()
        y2 = plt_df['last'].to_numpy()
        y1 = np.diff(y1)
        y2 = np.diff(y2)
        # xnorm = plt_df['monthfraction'].to_numpy()
        x_week = plt_df['dayofweek'] / 7 + plt_df['dayfraction'] / 7
        x_week = np.diff(x_week)
        # print("min x diff", np.abs(x_week).min(), np.abs(x_week).max())
        x_week = np.clip(np.abs(x_week), 1e-3, 3)

        plt.figure(figsize=(25, 10), dpi=400)
        # plt.plot(y1, label="open", alpha=0.5, color='green')
        # plt.plot(y2, label="close", alpha=0.3, color='red')
        plt.plot(plt_df['last'].values, label="last", alpha=0.3, color='red')
        # plt.plot(y2 / x_week / 2000, label="close speed", alpha=0.4, color='blue')
        # plt.plot(x_week, label="xweek", alpha=0.5, color='black')

        # y_wsk = plt_df['obv length'].to_numpy() / 3000
        # plt.plot(y_wsk, label="obv", alpha=0.7, color='black')

        plt.legend()
        plt.tight_layout()
        plt.savefig("openclose.png")

    "SAVE AND PRINT"
    # print(df.columns)
    # print(df.head())

    # plt.figure()
    # plt.plot(df['last'])
    # plt.show()

    if save_path:
        df.to_csv(os.path.join(path_data_clean_folder, "cleaned.csv"), index=False)

    return df


DATA_NORM_DICT = {
        "divide price": {
                "method": "div", "value": 5000,
                "keys": [
                        'open', 'high', 'low', 'last',
                        'top band', 'middle band', 'bottom band',
                        'ohlc avg', 'hlc avg', 'hl avg'
                ]
        },
        "NoTrades": {
                "method": "div", "value": 500,
                "keys": [
                        '# of trades',
                ]
        },
        "Non Zero": {
                "method": "div", "value": 500,
                "keys": [
                        'nonzero bid&ask vol at high/low highlight & extension lines',
                ]
        },
        "Std Var": {
                "method": "stdvar", "value": (301, 77),
                "keys": [
                        'bid volume', 'ask volume',
                ]
        },
        "drop keys": {
                "method": "drop", "value": '',
                "keys": [
                        'index', 'volume', 'dayofweek', 'daysinmonth', 'monthfraction',
                        'price at minimum highlight',
                        'date', 'time', 'timestamp_str', 'seconds',
                        'weekno',
                ]
        },
        "obv": {
                "method": "div", "value": 6000,
                "keys": ['obv length']
        },
}


def normalize(df):
    # div_price_keys = [
    #         'open', 'high', 'low', 'last',
    #         'top band', 'middle band', 'bottom band',
    #         'ohlc avg', 'hlc avg', 'hl avg'
    # ]

    # drop_keys =
    for name, norm_dc_params in DATA_NORM_DICT.items():
        df[norm_dc_params['keys']]  # Check Key in df
        mth = norm_dc_params['method']
        if mth == "div":
            val = float(norm_dc_params['value'])
            # print(name, "Divide", val, norm_dc_params['keys'])
            df[norm_dc_params['keys']] = df[norm_dc_params['keys']].astype(float) / val
            # print(df[norm_dc_params['keys']])

        elif mth == "minmax":
            sub, dv = float(norm_dc_params['value'][0]), float(norm_dc_params['value'][1])
            df[norm_dc_params['keys']] = (df[norm_dc_params['keys']] - sub) / dv

        elif mth == "stdvar":
            mu, std = float(norm_dc_params['value'][0]), float(norm_dc_params['value'][1])
            # print(name, "StdVar", mu, std, norm_dc_params['keys'])
            df[norm_dc_params['keys']] = (df[norm_dc_params['keys']] - mu) / std

        elif mth == "drop":
            # print(name, "Dropping", norm_dc_params['keys'])
            df = df.drop(columns=norm_dc_params['keys'])
        else:
            print(f"Invalid method: {mth}")
            raise ValueError(f"Invalid norm method: {mth}")

    "Check keys"

    return df


def to_sequences_1d(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size - 1):
        # print(i)
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)


def to_sequences_2d(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size - 1):
        # print(i)
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)


# @measure_real_time_decorator


from common_functions import interp_1d_sub


def interpolate_segments(segments_list, int_interval_s=1, include_time=False):
    out_list = []
    out_columns = []
    # print("OVB is here")
    # print(f"Interpolating: {len(segments_list)} segments.")

    for segment in segments_list:
        # print(segment)
        # print(segment['timestamp_ns'])
        tm_s = (segment['timestamp_ns'].to_numpy() / 1e9)

        segment = segment.drop(columns=['timestamp_ns'])
        tm_uniform = np.arange(tm_s[0], tm_s[-1] + int_interval_s, int_interval_s, dtype=int)
        # print(f"Tm uniform: {tm_uniform}")

        if include_time:
            columns = ['timestamp_s']
            out_array = np.array(tm_uniform).reshape(-1, 1)
            # print("Array from timestamp", out_array.dtype)
        else:
            out_array = None
            columns = []

        for c_i, col in enumerate(segment):
            vals = segment[col].values
            vals_uni = interp_1d_sub(tm_uniform, tm_s, vals).reshape(-1, 1)

            if out_array is None:
                out_array = vals_uni
            else:
                out_array = np.concatenate([out_array, vals_uni], axis=1)
            columns.append(col)

        # print(vals_uni.shape, tm_uniform.shape, tm_s.shape)
        out_list.append(out_array)
        out_columns.append(np.array(columns))

    return out_list, out_columns


def generate_interpolated_data(
        dataframe=None, csv_path=None, interval_s=1,
        include_time=True, split_interval_s=1800,
):
    """"""
    if dataframe is None:
        "Load df"
        if csv_path is None:
            raise ValueError("Must use `dataframe` or `csv_path` keyparam.")
        df = pd.read_csv(csv_path)
    else:
        df = dataframe

    "Normalize data"
    df = normalize(df)

    "SPLIT INTO SEGMENTS"
    if split_interval_s <= 0:
        segments = [df]
    else:
        segments = split_df_to_segments(df, split_s=split_interval_s, minimum_samples_per_segment=5)

    "INTERPOLATE SEGMENTS"
    segments_uni, columns = interpolate_segments(
            segments,
            int_interval_s=interval_s,
            include_time=include_time
    )
    # print("Columns:")
    # print(columns)
    del segments

    # print(segments_uni[0][0, :])
    return segments_uni, columns


@measure_real_time_decorator
def preprocess_pipe_uniform(
        input_data_path, interval_s=10,
        include_time=False, split_interval_s=1800,
        add_timediff_feature=False,
):
    dataframe = pd.read_csv(input_data_path)

    "CLEAN"
    dataframe = preprocess(dataframe)

    "NORMALIZE"
    segments, columns = generate_interpolated_data(
            dataframe=dataframe, include_time=True,
            interval_s=interval_s, split_interval_s=split_interval_s
    )

    "MORE FEATURES"
    if add_timediff_feature:
        tmps_ind = np.argwhere(columns[0] == "timestamp_s").ravel()[0]
        for i, (segm, col) in enumerate(zip(segments, columns)):
            tmps = segm[:, tmps_ind]
            diff_m = np.abs(np.diff(tmps) / 60)
            diff_m = np.concatenate([[0], diff_m]).reshape(-1, 1)
            segm = np.concatenate([segm, diff_m], axis=1)
            segments[i] = segm
            col = np.concatenate([col, ['timediff_m']])
            columns[i] = col

    if not include_time:
        # print("Pipe: Removing timestamp from all segments.")
        tmps_ind = np.argwhere(columns[0] == "timestamp_s").ravel()[0]
        # print("Segments pre:", segments[0].shape)
        segments = [np.delete(segm, tmps_ind, axis=1) for segm in segments]
        # print("Segments post:", segments[0].shape)
        # print(f"Columns pre: {columns[0]}")
        columns = [np.delete(col, tmps_ind, axis=0) for col in columns]
        # print(f"Columns post: {columns[0]}")

    return segments, columns


def convert_bars_to_traindata(
        list_ofdf,
        bars_n=10,
        add_time_diff=True,
        use_val_diff=False,
        workers=6,
):
    train_segments = []
    train_columns = []
    args = []
    for segi, segment_df in enumerate(list_ofdf):
        args.append((segi, segment_df, bars_n, add_time_diff, use_val_diff))

    pool = mpc.Pool(workers)
    result = pool.map(convert_thread, args)

    for res in result:
        if res is None:
            continue
        seq, col = res
        train_segments.append(seq)
        train_columns.append(col)
    # all_samples = sum(map(len, train_segments))
    # print(f"All samples: {all_samples} in: {len(train_segments)} segments")
    # print(segm_columns, len(segm_columns))
    #

    return train_segments, train_columns


def convert_thread(args):
    segi, segment_df, bars_n, add_time_diff, use_val_diff = args
    segment_df = segment_df.copy()
    print(f"Segment {segi:>3}: {segment_df.shape}. cols: ")
    # print(segment_df.columns)

    timestamp_ind = np.argwhere(segment_df.columns == "timestamp_ns").ravel()[0]
    segment_df.iloc[:, timestamp_ind] = segment_df.iloc[:, timestamp_ind] / 1e9
    # print(f"timestamp_ind: {timestamp_ind}")

    base_features = segment_df.shape[1]
    # segm_columns = segment_df.columns.to_numpy()
    segm_columns = np.array(segment_df.columns)
    # print(f"Pre Segment: {segm_columns}")
    segm_columns[timestamp_ind] = "timestamp_s"
    # print(f"Post Segment: {segm_columns}")
    # continue

    if len(segment_df) < bars_n:
        print(f"Skipping short segment: {segment_df.shape}")
        return
    elif (use_val_diff or add_time_diff) & (len(segment_df) <= bars_n):
        print(f"Skipping short segment +1: {segment_df.shape}")
        return

    if use_val_diff or add_time_diff:
        offset = 1
        if add_time_diff:
            base_features += 1
            segm_columns = np.concatenate([segm_columns, ['timediff_s']])

    else:
        offset = 0

    sequences_3d = np.empty((0, bars_n, base_features), dtype=float)

    for sample in range(offset, len(segment_df) - bars_n + 1):
        if use_val_diff:
            step_df_2d = segment_df.iloc[sample - 1:sample + bars_n].to_numpy()
            step_df_2d = np.diff(step_df_2d, axis=0)
            # print(f"Step diff: {step_df_2d.shape}")
            # print(step_df_2d)
        else:
            step_df_2d = segment_df.iloc[sample:sample + bars_n].to_numpy()

        if add_time_diff:
            timestamps = segment_df.iloc[sample - 1:sample + bars_n, timestamp_ind]
            stamp_diff_s = np.diff(timestamps).reshape(-1, 1)
            # print(f"Stamp: {stamp_diff_s.shape}, stepdf: {step_df_2d.shape}")
            step_df_2d = np.concatenate([step_df_2d, stamp_diff_s], axis=1)

        # print(f"step: {step_df_2d.shape}")
        # step_df_2d[np.newaxis, :]
        sequences_3d = np.concatenate([sequences_3d, step_df_2d[np.newaxis, :]], axis=0)
        # print(f"seq 3d: {sequences_3d.shape}")

        # print(f"Scope: {sample}:{sample + bars_n} / {len(segment_df)}: size: {step_df_2d.shape}")
        # print(step_df_2d.shape, )
    # print(f"End seq 3d: {sequences_3d.shape}")
    # print(f"End cols: {segm_columns}")
    # train_segments.append(sequences_3d)
    # train_columns.append(segm_columns)
    # pass
    return sequences_3d, segm_columns


@measure_real_time_decorator
def preprocess_pipe_bars(
        input_data_path,
        get_n_bars=10,
        use_bars_diff=False,
        history_interval=10,
        include_timestamp=False,
        split_interval_s=1800,
        add_timediff_feature=True,
        first_sample_date=None,
        clip_dataframe=None,
        workers=6,
):
    """

    Args:
        input_data_path:
        get_n_bars:
        use_bars_diff:
        history_interval:
        include_timestamp:
        split_interval_s:
        add_timediff_feature:
        first_sample_date: 'YYYY-MM-DD'
            Do not 0 pad!
            "2023-6-15"
    Returns:

    """
    dataframe = pd.read_csv(input_data_path)

    "CLEAN"
    dataframe = preprocess(dataframe, first_sample_date=first_sample_date)

    if clip_dataframe:
        dataframe = dataframe.iloc[:clip_dataframe, :]

    "NORMALIZE"
    dataframe = normalize(dataframe)

    "SEGMENTS"
    # print(f"Input dataframe: {dataframe.shape}")
    list_dfsegments = split_df_to_segments(
            dataframe, split_s=split_interval_s,
            minimum_samples_per_segment=get_n_bars,
    )

    segments, columns = convert_bars_to_traindata(
            list_dfsegments,
            bars_n=get_n_bars,
            add_time_diff=add_timediff_feature,
            workers=workers,
    )

    "MORE FEATURES"

    if not include_timestamp:
        # print("Pipe: Removing timestamp from all segments.")
        tmps_ind = np.argwhere(columns[0] == "timestamp_s").ravel()[0]
        # print("Segments pre:", segments[0].shape)
        segments = [np.delete(segm, tmps_ind, axis=2) for segm in segments]
        # print("Segments post:", segments[0].shape)
        # print(f"Columns pre: {columns[0].shape}, {columns[0]}")
        columns = [np.delete(col, tmps_ind, axis=0) for col in columns]
        # print(f"Columns post: {columns[0].shape}, {columns[0]}")

    return segments, columns


# @measure_real_time_decorator
def split_df_to_segments(dataframe, split_s=1800, minimum_samples_per_segment=10):
    # print(f"Spliting dataframe: {dataframe.shape}")

    diff_s = np.abs(np.diff(dataframe['timestamp_ns'] / 1e9))
    nodes = np.argwhere(diff_s >= split_s).ravel()

    # print(f"Split data into: {len(nodes)} pieces: {nodes}")
    # print(nodes)

    if len(nodes) <= 0:
        return [dataframe]
    else:
        nodes = [0, *(nodes + 1)]

    segments = []
    for start, stop in zip(nodes, nodes[1:]):
        segm = dataframe.iloc[start:stop, :]
        if len(segm) >= minimum_samples_per_segment:
            segments.append(segm)
            # print(f"Adding segment: {start}-{stop}")
        # else:
        #     print(f"Not adding segment: {start}-{stop}")

    if len(nodes) <= 0 and len(dataframe) >= minimum_samples_per_segment:
        segments = [dataframe]

    elif nodes[-1] != len(dataframe):
        last_segm = dataframe.iloc[nodes[-1]:, :]
        if len(last_segm) >= minimum_samples_per_segment:
            segments.append(dataframe.iloc[nodes[-1]:, :])
            # print(f"Adding last segment: ({nodes[-1]}:)")
        # else:
        #     print(f"Not adding last segment: {nodes[-1]}:")
    # else:
    #         print("All fine.")

    return segments


if __name__ == "__main__":
    import sys


    os.makedirs(path_data_clean_folder, exist_ok=True)
    use('ggplot')
    # np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf, formatter=dict(int='{:d}'.format))

    "CLEAN"
    input_data_path = folder_danych + "on_balance_volume.txt"
    # dataframe = pd.read_csv(input_data_path)
    # dataframe = preprocess(dataframe, first_sample_date='2023-6-15')
    # dataframe = dataframe.iloc[800:1000, :]
    # print(dataframe.shape)
    #
    # x = (dataframe['timestamp_ns'] / 1e9).to_numpy()
    # x -= x[0]
    # y = dataframe['last']
    #
    # plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # plt.subplot(3, 1, 1)
    # plt.plot(x, y)
    #
    # plt.subplot(3, 1, 2)
    # diff = np.diff(y)
    # plt.plot(x[1:], diff)
    #
    # plt.subplot(3, 1, 3)
    # xdiff = np.diff(x)
    # plt.plot(x[1:], diff / xdiff)
    #
    # plt.show()


    # for num in dataframe.loc[:5, 'timestamp_ns'].to_numpy() / 1e9:
    #     print(num)

    # segments1 = split_df_to_segments(dataframe, 1800)
    # print(len(segments1))
    # segments, columns = preprocess_pipe_uniform(input_data_path, split_interval_s=1800,
    #                                             include_time=True)
    segments, columns = preprocess_pipe_bars(
            input_data_path,
            get_n_bars=80,
            split_interval_s=1800,
            include_timestamp=True,
            first_sample_date="2023-6-27"
    )
    # print(segments[])
    # first = segments[0]
    # first = first.astype(int)
    # print(first)
    # print(first[:5, 0])

    # from common_functions import to_sequences_forward


    # train_segments, _ = to_sequences_forward(segments[0], 10, [1])
    # train_segments = train_segments.astype(int)

    # print("RES:")
    # print(train_segments[0].shape)
    # print(train_segments[:5, :5, 0])

    # print(len(segments))
    # print(columns[0], type(columns[0]))
    #
    # plt.plot(segments1[0]['last'])
    # plt.title("Interpolated")
    # plt.figure()
    # price_ind = np.argwhere(columns[0] == "last").ravel()[0]
    # print(f"price ind: {price_ind}")
    # plt.plot(segments[0][:, price_ind])
    # plt.show()
    # for i, seg in enumerate(segments):
    #     print(i, len(seg))
