import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from datetime import timedelta


folder_danych = os.path.abspath(os.path.join("..", "dane", "")) + os.path.sep
folder_danych_clean = os.path.abspath(os.path.join("..", "dane", "clean", "")) + os.path.sep


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


def preprocess(df, firstday_index=8895, allow_plot=False):
    print("Processing data")

    "Discard bad days"
    df = df.iloc[firstday_index:, :]
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

        y_wsk = plt_df['obv length'].to_numpy() / 3000
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

    df.to_csv(os.path.join("..", "dane", "clean", "test.csv"), index=False)


NORM_DICT = {
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
        "NonZero": {
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
    for name, norm_dc_params in NORM_DICT.items():
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


def to_sequences_forward(array, seq_size=1, fwd_intervals=[1]):
    x = []
    y = []
    offset_arr = np.array(fwd_intervals) - 1
    last_minus = max(fwd_intervals)-1
    for i in range(len(array)  - seq_size - last_minus):
        window = array[i:(i + seq_size), :]
        x.append(window)
        sub_arr = array[i + seq_size + offset_arr, :]
        y.append(sub_arr)
    # print("Returning:")
    # print(x)
    # print(y)
    return np.array(x), np.array(y)


from common_functions import interp_2d, interp_1d_sub


def interpolate_segments(segments_list, int_interval_s=1):
    out_list = []
    print("OVB is here")

    for segment in segments_list:
        tm_s = (segment['timestamp_ns'] / 1e9).values
        # diff = tm_s[-1] - tm_s[0]
        # print(tm_s[0], tm_s[-1], diff)
        # print("segment", diff)
        # print(segment.columns)
        segment = segment.drop(columns=['timestamp_ns'])
        tm_uniform = np.arange(tm_s[0], tm_s[-1] + int_interval_s, int_interval_s)

        out_array = np.array(tm_uniform).reshape(-1, 1)

        for col in segment:
            print(f"Interpolating: {col}")
            vals = segment[col].values
            vals_uni = interp_1d_sub(tm_uniform, tm_s, vals).reshape(-1, 1)
            out_array = np.concatenate([out_array, vals_uni], axis=1)

        # print(vals_uni.shape, tm_uniform.shape, tm_s.shape)
        out_list.append(out_array)
    return out_list


def generate_data(csv_path):
    """"""
    "Load df"
    df = pd.read_csv(csv_path)

    "End week indexes"
    split_week_mask = df.loc[:len(df) - 2, 'dayofweek'].values > df.loc[1:, 'dayofweek'].values
    split_week_inds = np.argwhere(split_week_mask).ravel() + 1

    # mu = df['non'].mean()
    # vals = (df['bid volume'] - mu)
    # std = vals.std()
    # vals = vals / std
    # print(vals)
    # print(mu)
    # print(std)

    "Normalize data"
    df = normalize(df)

    df.to_csv(folder_danych_clean + "normalized.csv", index=False)

    segments = [df]
    # inds = np.concatenate([[0], split_week_inds, [len(df)]])
    # for start, stop in zip(inds, inds[1:]):
    #     week_segment = df.iloc[start:stop, :]
    #     segments.append(week_segment)

    segments_uni = interpolate_segments(segments, int_interval_s=1)
    del segments

    print(segments_uni[0][0, :])


if __name__ == "__main__":
    # input_data_path = folder_danych + "on_balance_volume.txt"
    # dataframe = pd.read_csv(input_data_path)
    # preprocess(dataframe)

    # print(folder_danych_clean)
    path = os.path.join(folder_danych_clean, "test.csv")
    generate_data(path)
