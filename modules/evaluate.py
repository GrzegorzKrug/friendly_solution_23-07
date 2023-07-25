import numpy as np

import time
from tensorflow import keras
import os
import datetime

from random import sample, shuffle
from actors import initialize_agents, resolve_actions_multibuy, resolve_actions_singlebuy

from common_settings import ITERATION, path_data_clean_folder, path_models
from common_functions import (
    NamingClass, get_splits, get_eps, to_sequences_forward, load_data_split,
    unpack_evals_to_table,
)
from reward_functions import RewardStore

from functools import wraps
from collections import deque
from model_creator import grid_models_generator, model_builder

from io import TextIOWrapper
from yasiu_native.time import measure_real_time_decorator

import traceback
import multiprocessing

import tensorflow as tf

import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.style import use


# session_dataframe.loc[
#     len(session_dataframe)] = session_eps, i_train_sess, ses_start, ses_end, g


def evaluate(
        model_keras: keras.Model, datalist_2dsequences_ordered_train,
        price_col_ind,
        naming_ob: NamingClass,
        # fulltrain_ntimes=1000,
        # agents_n=10,
        session_size=3600,

        # Optional
        max_eps=0.5, override_eps=None,
        remember_fresh_fraction=0.2,
        train_from_oldmem_fraction=0.2,

        reward_f_num=3,
        game_n=3,
        # discount=0.9,
        # director_loc=None, name=None, timeout=None,
        # save_qval_dist=False,
        # retrain_from_all=1,

        # PARAMS ==============
        # time_window_size=10,
        # stock_price_multiplier=0.5,
        # stock_ammount_in_bool=True,
        # reward_fun: reward_fun_template,
        allow_multibuy=False,

        # FILE SAVERS
        # qvals_file: TextIOWrapper = None,
        # session_file: TextIOWrapper = None,
        # time_file: TextIOWrapper = None,
        # loss_file: TextIOWrapper = None,
        # rew_file: TextIOWrapper = None,
):
    # RUN_LOGGER.debug(
    #         f"Train params: {naming_ob}: trainN:{fulltrain_ntimes}, agents: {agents_n}. Reward F:{reward_f_num}")
    N_SAMPLES = len(datalist_2dsequences_ordered_train)
    WALK_INTERVAL_DEBUG = 250
    # RUN_LOGGER.debug(f"Input samples: {datalist_2dsequences_ordered_train.shape}")

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    "GET MODEL PATH"
    path_this_model_folder = os.path.join(path_models, naming_ob.path, "")
    if os.path.isfile(path_this_model_folder + "weights.keras"):
        # RUN_LOGGER.info(f"Loading model: {path_this_model_folder}")
        model_keras.load_weights(path_this_model_folder + "weights.keras")
    else:
        print(f"Not found model for evaluation: {naming_ob.path}")
        return
        # raise FileNotFoundError(f"Not found model: {path_this_model_folder}")
        # RUN_LOGGER.info("Not loading model.")

    if allow_multibuy:
        resolve_actions_func = resolve_actions_multibuy
    else:
        resolve_actions_func = resolve_actions_singlebuy

    if N_SAMPLES <= 0:
        raise ValueError(
                f"Too few samples! {N_SAMPLES}, shape:{datalist_2dsequences_ordered_train.shape}")

    # reward_fun = RewardStore.get(reward_f_num)
    # out_size = int(naming_ob.outsize)
    # qv_dataframe = pd.DataFrame(columns=['eps', 'sess_i', 'sample_n', 'buy', 'idle', 'sell'])
    # session_dataframe = pd.DataFrame(columns=['eps', 'session_num', 'ind_start', 'ind_end', 'gain'])

    LOOP_TIMES = deque(maxlen=100)
    agents_n = 1
    use('ggplot')
    colors = (
            (0, 0, 0),
            (0, 0.4, 0.8),  # cash
            (0.6, 0.3, 0),  # cargo
            (0.2, 0.2, 0.4),  # action
            (0, 0.8, 0.2),  # gain
            (0.6, 0, 0),  # q3
            (0, 0, 0.6),  # q2
            (0, 0.6, 0),  # q1
            (1, 0, 0),  # price
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
    )

    if game_n <= 0:
        starts = [0]
    elif game_n == 3:
        starts = [0, 500, 1000]
    elif game_n == 4:
        starts = [0, 500, 1000, 1500]
    elif game_n == 5:
        starts = [0, 500, 1000, 1500, 3000]
    elif game_n == 7:
        starts = [0, 500, 1000, 1500, 3000, 4000, 5000]
    else:
        starts = [0]
        print(f"Unsuported games amount: {game_n}")
        # raise ValueError(f"Not supported amount of games: {game_n}")

    eval_values = []

    for i_train_sess, ses_start in enumerate(starts):
        # last_start = N_SAMPLES - session_size
        # if i_train_sess == 0:
        #     ses_start = 0
        # else:
        #     ses_start = np.random.randint(0, last_start)
        ses_end = ses_start + session_size

        agents_discrete_states, hidden_states = initialize_agents(agents_n)

        "Actions:"
        "0, 1, 2"
        "Sell, Pass, Buy"
        starttime = time.time()

        how_many_actions = 0
        how_many_valid = 0

        labels = ['i_sample', 'Cash', 'Cargo', 'Action', 'Gain', 'Q1', 'Q2', 'Q3', 'price']
        plot_array = np.zeros((0, len(labels)), dtype=float)
        # print()
        # print(f"Session: {i_train_sess}")
        # print(f"Walking start sample: {ses_start}")

        for i_sample in range(ses_start, ses_end - 1):  # Never in done state
            # print(f"Sample: {i_sample}")
            done_session = i_sample == (N_SAMPLES - 1)  # is this last sample?

            timesegment_2d = datalist_2dsequences_ordered_train[i_sample, :]
            timesegment_stacked = np.tile(timesegment_2d[np.newaxis, :, :], (agents_n, 1, 1))

            # if not i_sample % WALK_INTERVAL_DEBUG:
            #     print(f"Walking sample: {i_sample}")

            "MOVE TO IF BELOW"
            # print(timesegment_stacked.shape)
            # print(agents_discrete_states.shape)
            # print(f"Price col: {price_col_ind}")

            q_vals = model_keras.predict(
                    [timesegment_stacked, agents_discrete_states],
                    verbose=False
            )
            actions = np.argmax(q_vals, axis=-1)

            if actions[0] in [0, 2]:
                how_many_actions += 1

            if actions[0] == 0 and hidden_states[0, 2] == 0:
                "Buy when 0"
                how_many_valid += 1
            elif actions[0] == 2 and hidden_states[0, 2] == 1:
                "Sell when 1"
                how_many_valid += 1

            # rewards = []
            # valids = []
            # env_state_arr = timesegment_2d

            # cur_step_price = env_state_arr[0, price_col_ind]
            cur_step_price = datalist_2dsequences_ordered_train[i_sample, 0, 3]
            # print(datalist_2dsequences_ordered_train.shape)
            # print(cur_step_price, type(cur_step_price))

            "Dont train"
            new_states, new_hidden_states = resolve_actions_func(
                    cur_step_price, agents_discrete_states, hidden_states, actions
            )
            step_gain = 0.3

            plot_vec = [
                    i_sample,
                    hidden_states[0, 0], hidden_states[0, 2], actions[0],
                    step_gain, *q_vals[0, :], cur_step_price,
            ]
            plot_vec = np.array(plot_vec).reshape(1, -1)

            # print(plot_array.shape, plot_vec.shape)
            plot_array = np.concatenate([plot_array, plot_vec], axis=0)

            agents_discrete_states = new_states
            hidden_states = new_hidden_states

        # tend_walking = time.time()
        # print(f"End cargo: {hidden_states[0, 2]} and price: {cur_step_price}")
        end_gain = hidden_states[0, 0] - hidden_states[0, 1] + hidden_states[0, 2] * cur_step_price

        plt.subplots(3, 1, figsize=(20, 10), dpi=200, height_ratios=[3, 2, 1])
        # gain = hidden_states[:, 0] - hidden_states[:, 1]
        x = plot_array[:, 0]

        plt.subplot(3, 1, 1)
        # for i, lb in enumerate(labels[1:3], 1):
        #     plt.plot(x, plot_array[:, i], label=lb, color=colors[i], alpha=0.8, linewidth=2)
        for act in [0, 1, 2]:
            mask = plot_array[:, 3] == act
            xa = x[mask]
            ya = plot_array[mask, -1]
            lb = {0: "Buy", 1: "Pass", 2: "Sell"}[act]
            s = {0: 35, 1: 15, 2: 40}[act]
            plt.scatter(xa, ya, label=f"Action: {lb}", s=s)

        # print()
        eval_values.append((how_many_actions, how_many_valid, np.round(end_gain, 5)))

        plt.plot(x, plot_array[:, -1], label="Price", color=colors[0], alpha=0.6, linewidth=2)
        plt.title("Price")
        plt.legend()

        plt.subplot(3, 1, 2)
        for i, lb in enumerate(labels[1:4], 1):
            if i == 3:
                plt.plot(x, plot_array[:, i] + 1, label=lb, color=colors[i], alpha=0.8, linewidth=2)
            else:
                plt.plot(x, plot_array[:, i], label=lb, color=colors[i], alpha=0.8, linewidth=2)
        plt.title("Actions and cash")
        plt.legend()

        plt.subplot(3, 1, 3)
        for i, lb in enumerate(labels[5:-1], 5):
            lb = {5: "Q1:Buy", 6: "Q2:Pass", 7: "Q3:Sell"}[i]
            plt.plot(x, plot_array[:, i], label=lb, color=colors[i], alpha=0.8, linewidth=2)

        plt.title("Q vals")
        plt.legend()

        plt.suptitle(naming_ob.path)
        plt.xlabel("sample number")
        plt.tight_layout()
        plt.savefig(os.path.join(path_this_model_folder, "data", f"eval_plot-{i_train_sess}.png"))
        plt.close()
        print(f"Saved fig: {naming_ob.path} - eval - {i_train_sess}")
    return naming_ob.path, eval_values

    # if allow_train:
    #     model_keras.save_weights(path_this_model_folder + "weights.keras")
    #     RUN_LOGGER.info(f"Saved weights: {naming_ob}")


    # return history, best, best_all


def single_model_evaluate(
        counter, model_params, train_sequences, price_id, game_n
):
    "LIMIT GPU BEFORE BUILDING MODEL"

    "GLOBAL LOGGERS"
    # DISCOUNT = 0  # .95

    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        (arch_num, time_feats, time_window, float_feats, out_size,
         nodes, lr, batch, loss, discount
         ) = model_params
        model = model_builder(
                arch_num,
                time_feats, time_window, float_feats, out_size,
                loss, nodes, lr
        )
        reward_fnum = 5

        naming_ob = NamingClass(
                arch_num, ITERATION,
                time_feats=time_feats, time_window=time_window, float_feats=float_feats,
                outsize=out_size,
                node_size=nodes, reward_fnum=reward_fnum,

                learning_rate=lr, loss=loss, batch=batch,
                discount=discount,
        )
    except Exception as exc:
        print(f"EXCEPTION when setting model: {exc}")
        return None

    try:
        # for gpu in tf.config.experimental.list_physical_devices("GPU"):
        #     pass
        # f"Limitig gpu: {gpu}"
        # tf.config.experimental.set_memory_growth(gpu, True)
        result = evaluate(
                model, train_sequences,
                price_col_ind=price_id,
                naming_ob=naming_ob,
                session_size=500,
                reward_f_num=reward_fnum,
                game_n=game_n,
        )
        return result
    except Exception as exc:
        "PRINT TO SYS"
        print(f"EXCEPTION during evaluation: {exc} {exc.__traceback__}")
        # print(traceback.print_last())
        text = '\n'.join(traceback.format_tb(exc.__traceback__, limit=None))
        print(text)
        print("^^^^")
        # stbbb = traceback.extract_tb(exc.__traceback__, limit=10)
        # for ftb in stbbb:
        #     print(ftb)
        # print(stbbb)

    "Clear memory?"
    del model


if __name__ == "__main__":
    use('ggplot')
    "LOAD Interpolated data"
    columns = np.load(path_data_clean_folder + "int_norm.columns.npy", allow_pickle=True)
    print(
            "Loading file with columns: ", columns,
    )
    price_col = np.argwhere(columns == "last").ravel()[0]
    print(f"Price `last` at col: {price_col}")
    train_data, test_data = load_data_split(path_data_clean_folder + "int_norm.arr.npy")

    time_wind = 10
    float_feats = 1
    out_sze = 3
    train_sequences, _ = to_sequences_forward(train_data[:7500, :], time_wind, [1])

    samples_n, _, time_ftrs = train_sequences.shape
    print(f"Train sequences shape: {train_sequences.shape}")

    "Model Grid"
    gen1 = grid_models_generator(time_ftrs, time_wind, float_feats=float_feats, out_size=out_sze)
    # gen1 = dummy_grid_generator()
    # for data in gen1:
    #     single_model_training_function(*data)
    game_n = 5

    with ProcessPoolExecutor(max_workers=6) as executor:
        process_list = []
        for counter, data in enumerate(gen1):
            proc = executor.submit(
                    single_model_evaluate, *data, train_sequences, price_col, game_n
            )
            process_list.append(proc)
            print(f"Adding eval model: {counter}")
            # if counter >= 4:
            #     break

            # while True
            # break
        # proc.e
        #     results.append(proc)
        print("Waiting:")
        # result = concurrent.futures.wait(process_list)
        # print("Waiting finished.")
        for proc in process_list:
            proc.result()

        print("All processes have ended...")

        results = [proc.result() for proc in process_list]
        print("results:")
        print(results)

        tab = unpack_evals_to_table(results, game_n)
        print(tab)
        now = datetime.datetime.now()
        dt_str = f"{now.day}.{now.month}-{now.hour}.{now.minute}"
        with open(os.path.join(path_models, f"evals-{dt_str}.txt"), "wt") as fp:
            fp.write(str(tab))

    # for res in process_list:
    #     print(res)
    #     res.join()
