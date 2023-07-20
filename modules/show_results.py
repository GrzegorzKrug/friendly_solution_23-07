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
    # sample_min


    plt.subplots(3, 1, figsize=(25, 12), height_ratios=[5, 2, 6], sharex=True)
    plt.subplot(3, 1, 1)
    plt.plot(x_sess, sess_df['gain'], label='EndGain', color='green')
    plt.plot(x_sess, sess_df['sess_eps'], label='Exploration', color='black', alpha=0.7)
    plt.plot(x_loss, loss_df['session_meanloss'], label='mean loss', color='red')
    plt.legend(markerscale=4)

    plt.subplot(3, 1, 2)
    plt.scatter(x_reward, rew_df['reward'], label='Rewards', alpha=0.2, s=5, color='blue')
    plt.legend(markerscale=4)

    plt.subplot(3, 1, 3)
    q_arr = qval_df.loc[:, ['q1', 'q2', 'q3']].to_numpy()

    # temp_x = qval_df['i_sample'].to_numpy().astype(float)
    # temp_x -= temp_x.min()
    # temp_x /= temp_x.max()
    # x_qvl = qval_df['i_train_sess'].to_numpy() + temp_x
    x_qvl = np.linspace(0, sess_df.loc[len(sess_df) - 1, 'i_train_sess'], len(qval_df))
    for qi, q in enumerate(q_arr.T):
        plt.scatter(x_qvl, q, label=f"Q{qi + 1}", s=5, alpha=0.8)
    plt.legend(markerscale=4)

    if naming:
        plt.suptitle(f"Model: {naming.path}")

    plt.xlim(-1, x_sess[-1] + 1)
    plt.xlabel("Epoch")
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