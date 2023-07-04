import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from datetime import timedelta


folder_danych = os.path.join("..", "dane", "")
folder_danych_clean = os.path.join("..", "dane", "clean", "")
path = folder_danych + "on_balance_volume.txt"
dataframe = pd.read_csv(path)


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


def preprocess(df, firstday='2023-6-15', allow_plot=True):
    firstday_index = 8895
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
    print(dt_64, dt_64.dtype)
    print(dt_64.dt.dayofweek)

    tdelta = df['time'].apply(timestr_to_timedelta)
    timestamp = dt_64.values + tdelta
    df = df.assign(timestamp_str=timestamp)
    df = df.assign(timestamp_us=timestamp.astype(np.int64))
    # print(timestamp.astype(int))

    # print(dir(dt_64))
    # print(timestamp)
    # ""
    # df = df.assign(timestamp=timestamp)

    "Day of week"
    df = df.assign(dayofweek=dt_64.dt.dayofweek)

    "Days in month"
    df = df.assign(daysinmonth=dt_64.dt.days_in_month)

    "Month fraction"
    df = df.assign(monthfraction=dt_64.dt.day / dt_64.dt.days_in_month)

    # dt = pd.DataFrame(dt_64)
    # print(dt)
    # print(dt['date'].dayofweek)

    if allow_plot:
        "No plot"
        plt_df = df.iloc[:1000]
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
        plt.plot(y2 / x_week / 2000, label="close speed", alpha=0.4, color='blue')
        # plt.plot(x_week, label="xweek", alpha=0.5, color='black')


        y_wsk = plt_df['obv length'].to_numpy() / 3000
        plt.plot(y_wsk, label="obv", alpha=0.7, color='black')

        plt.legend()
        plt.tight_layout()
        plt.savefig("openclose.png")

    "SAVE AND PRINT"
    # print(df.columns)
    # print(df.head())
    df.to_csv(os.path.join("..", "dane", "clean", "test.csv"), index=False)


if __name__ == "__main__":
    preprocess(dataframe)
