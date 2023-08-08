import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common_functions import NamingClass
from common_settings import path_models

import traceback
import glob
import datetime
import os


folders = glob.glob(f"{path_models}*")
folders = [fold for fold in folders if os.path.isdir(fold)]

from matplotlib.style import use
import multiprocessing as mpc


use('ggplot')


# print(folders)

def make_plot(folder, dt_str, naming: NamingClass = None):
    ROWS = 4
    loss_df = pd.read_csv(os.path.join(folder, f"{dt_str}-loss.csv"))
    rew_df = pd.read_csv(os.path.join(folder, f"{dt_str}-rewards.csv"))
    qval_df = pd.read_csv(os.path.join(folder, f"{dt_str}-qvals.csv"))
    sess_df = pd.read_csv(os.path.join(folder, f"{dt_str}-sess.csv"))

    x_sess = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(sess_df))
    x_reward = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(rew_df))
    x_loss = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(loss_df))
    # sample_min

    plt.subplots(ROWS, 1, figsize=(25, 12), dpi=300, height_ratios=[2, 3, 2, 4], sharex=True)
    plt.subplot(ROWS, 1, 1)
    plt.plot(x_sess, sess_df['sess_eps'], label='Exploration', color='black', alpha=0.7)

    if "fresh_loss" in loss_df:
        plt.plot(x_loss, loss_df['fresh_loss'], label='Fresh loss', color='red')
        old_mem = loss_df['oldmem_loss']
        mask = old_mem >= 0
        x_oldmem = x_loss[mask]
        old_mem = old_mem[mask]
        if len(old_mem) > 2:
            plt.plot(x_oldmem, old_mem, label='Oldmem loss', color='brown')
    else:
        plt.plot(x_loss, loss_df['session_meanloss'], label='Mean loss', color='red')

    plt.legend(loc='upper left', markerscale=4)

    plt.subplot(ROWS, 1, 2)
    plt.scatter(x_reward, rew_df['reward'], label='Rewards', alpha=0.2, s=5, color='blue')
    plt.legend(loc='upper left', markerscale=4)

    plt.subplot(ROWS, 1, 3)
    q_arr = qval_df.loc[:, ['q1', 'q2', 'q3']].to_numpy()

    # temp_x = qval_df['i_sample'].to_numpy().astype(float)
    # temp_x -= temp_x.min()
    # temp_x /= temp_x.max()
    # x_qvl = qval_df['i_train_sess'].to_numpy() + temp_x
    x_qvl = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(qval_df))
    for qi, q in enumerate(q_arr.T):
        plt.scatter(x_qvl, q, label=f"Q{qi + 1}", s=5, alpha=0.6)
    plt.legend(loc='upper left', markerscale=4)

    plt.subplot(ROWS, 1, 4)
    plt.plot(x_sess, sess_df['gain'], label='EndGain', color='green')
    plt.plot(x_sess[[0, -1]], [0, 0], color='black')
    plt.legend()

    if naming:
        plt.suptitle(f"Model: {naming.path}")

    plt.xlim(-1, x_sess[-1] + 1)
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"sess-{dt_str}.png"))
    plt.close()
    # print(sess_df)


def plot_thread(args):
    cur_model_path, dt_str, naming = args
    try:
        make_plot(os.path.join(cur_model_path, "data"), dt_str, naming)
        print(f"Plotted: {naming.path} - {dt_str}")
    except Exception as exc:
        print(f"Can not plot: {cur_model_path}- {dt_str}\n error: ({exc})")
        text = '\n'.join(traceback.format_tb(exc.__traceback__, limit=None))
        print(text)


if __name__ == "__main__":
    args = []
    plot_date_cutoff = datetime.datetime.now() - datetime.timedelta(days=2)
    print(f"date: {plot_date_cutoff}")

    for cur_model_path in folders:
        # name = os.path.basename(cur_model_path)
        # print(f"name: {name}")
        try:
            naming = NamingClass.from_path(cur_model_path)
        except IndexError as exc:
            print(f"Skipping results of folder: {cur_model_path} ")
            continue
        # print(naming)

        res_files = glob.glob(os.path.join(cur_model_path, "data", "") + "*")
        file_names = [os.path.basename(fil) for fil in res_files]
        dates = set()
        for file in file_names:
            if not file.endswith('.csv'):
                continue
            dt, tm, nm = file.split('-')
            dt_str = f"{dt}-{tm}"
            dates.add(dt_str)

        # print(f"Adding args: {naming.path}")
        for dt_str in dates:
            # plot_thread()
            # print(dt_str)
            dt_ob = datetime.datetime.strptime(dt_str, "%Y.%m.%d-%H.%M")
            if plot_date_cutoff <= dt_ob:
                print(f"Adding: {cur_model_path} - {dt_str}")
                # print(dt_str<first_plot)
                # print(f"Adding: {cur_model_path}")
                args.append((cur_model_path, dt_str, naming))

    pool = mpc.Pool(6)
    pool.map(plot_thread, args)
