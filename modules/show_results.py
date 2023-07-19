import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common_functions import NamingClass
from common_settings import path_models

import glob
import os


folders = glob.glob(f"{path_models}*")
folders = [fold for fold in folders if os.path.isdir(fold)]

from matplotlib.style import use


use('ggplot')


# print(folders)

def make_plot(folder, dt_str, naming: NamingClass = None):
    loss_df = pd.read_csv(os.path.join(folder, f"{dt_str}-loss.csv"))
    rew_df = pd.read_csv(os.path.join(folder, f"{dt_str}-rewards.csv"))
    qval_df = pd.read_csv(os.path.join(folder, f"{dt_str}-qvals.csv"))
    sess_df = pd.read_csv(os.path.join(folder, f"{dt_str}-sess.csv"))

    print(sess_df)
    print(sess_df.shape)

    x_sess = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(sess_df))
    x_reward = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(rew_df))
    x_loss = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(loss_df))
    x_qvl = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(qval_df))

    plt.subplots(3, 1, figsize=(25, 12), height_ratios=[6, 1, 3])
    plt.subplot(3, 1, 1)
    plt.plot(x_sess, sess_df['gain'], label='EndGain', color='green')
    plt.plot(x_sess, sess_df['sess_eps'], label='Exploration', color='black', alpha=0.7)
    plt.scatter(x_reward, rew_df['reward'], label='Rewards', alpha=0.2, s=5, color='blue')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x_loss, loss_df['session_meanloss'], label='mean loss', color='red')
    plt.legend()

    plt.subplot(3, 1, 3)
    q_arr = qval_df.loc[:, ['q1', 'q2', 'q3']].to_numpy()
    # q_args = np.argmax(q_arr, axis=1)
    # q_args = qval_df['']
    for q in q_arr.T:
        plt.plot(x_qvl, q)
    # plt.scatter(x_qvl, q_args, label="Q Decisions")

    if naming:
        plt.suptitle(f"Model: {naming.path}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"sess-{dt_str}.png"))
    # print(sess_df)


for cur_model_path in folders:
    # name = os.path.basename(cur_model_path)
    # print(f"name: {name}")
    naming = NamingClass.from_path(cur_model_path)
    # print(naming)

    res_files = glob.glob(os.path.join(cur_model_path, "data", "") + "*")
    # print(res_files)
    file_names = [os.path.basename(fil) for fil in res_files]
    # print(file_names[0])
    dates = set()
    for file in file_names:
        if not file.endswith('.csv'):
            continue
        dt, tm, nm = file.split('-')
        dt_str = f"{dt}-{tm}"
        dates.add(dt_str)

    for dt_str in dates:
        # print(f"Trying: {dt_str}")
        make_plot(os.path.join(cur_model_path, "data"), dt_str, naming)
