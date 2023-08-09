import numpy as np

import time
from tensorflow import keras
import os
import datetime

from random import sample, shuffle
from actors import initialize_agents, resolve_actions_multibuy, resolve_actions_singlebuy

from common_settings import ITERATION, path_data_clean_folder, path_models, path_data_folder
from common_functions import (
    NamingClass, get_splits, get_eps, to_sequences_forward, load_data_split,
    unpack_evals_to_table,
)
from reward_functions import RewardStore

from functools import wraps
from collections import deque
from model_creator import (
    model_builder,
    grid_models_generator, grid_models_generator_2,
    grid_models_generator_it23,
)

from io import TextIOWrapper
from yasiu_native.time import measure_real_time_decorator

import traceback
import multiprocessing

import tensorflow as tf

import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.style import use

from preprocess_data import preprocess_pipe_uniform, preprocess_pipe_bars
import gc


# session_dataframe.loc[
#     len(session_dataframe)] = session_eps, i_train_sess, ses_start, ses_end, g


def eval_func(
        model_keras: keras.Model, segments_oftraindata,
        price_col_ind,
        naming_ob: NamingClass,
        # fulltrain_ntimes=1000,
        # agents_n=10,
        session_size=3600,
        name="",

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
        timestamp_col=None,
        full_eval=False,

        # FILE SAVERS
        # qvals_file: TextIOWrapper = None,
        # session_file: TextIOWrapper = None,
        # time_file: TextIOWrapper = None,
        # loss_file: TextIOWrapper = None,
        # rew_file: TextIOWrapper = None,
        # time_sequences=None,
):
    # RUN_LOGGER.debug(
    #         f"Train params: {naming_ob}: trainN:{fulltrain_ntimes}, agents: {agents_n}. Reward F:{reward_f_num}")
    # WALK_INTERVAL_DEBUG = 250
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
    path_eval_folder = os.path.join(path_this_model_folder, "evals", "")
    os.makedirs(path_eval_folder, exist_ok=True)

    if allow_multibuy:
        resolve_actions_func = resolve_actions_multibuy
    else:
        resolve_actions_func = resolve_actions_singlebuy

    # reward_fun = RewardStore.get(reward_f_num)
    # out_size = int(naming_ob.outsize)
    # qv_dataframe = pd.DataFrame(columns=['eps', 'sess_i', 'sample_n', 'buy', 'idle', 'sell'])
    # session_dataframe = pd.DataFrame(columns=['eps', 'session_num', 'ind_start', 'ind_end', 'gain'])

    LOOP_TIMES = deque(maxlen=100)
    AGENTS_N = 1
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

    eval_values = []
    if full_eval:
        game_n = len(segments_oftraindata)
        print(f"Full eval, games: {game_n}")
        random_start = False

    elif len(segments_oftraindata) < game_n:
        random_start = True
    else:
        random_start = False

    # print(f"Segments: {len(segments_oftraindata)}, games: {game_n}, random: {random_start}")

    # random_start = True
    # else:
    # if game_n > len(segments_oftraindata):
    #     game_n = len(segments_oftraindata)
    #     print(f"Staring evaluation, games at max size: {game_n}")
    # else:
    #     print(f"Staring evaluation, games: {game_n} (of {len(segments_oftraindata)} segments)")

    for i_eval_sess in range(game_n):
        segm_i = np.linspace(0, len(segments_oftraindata) - 1, game_n).round().astype(int)[i_eval_sess]
        # print(f"Games: {game_n} at: {np.linspace(0, len(segments_oftraindata) - 1, game_n).round()}")

        ordered_list_of3dsequences = segments_oftraindata[segm_i]

        if timestamp_col is not None:
            time_sequences = ordered_list_of3dsequences[:, -1, timestamp_col]
            ordered_list_of3dsequences = np.delete(ordered_list_of3dsequences, timestamp_col, axis=2)
            # print(f"Removing time col: {timestamp_col}, new shape: {ordered_list_of3dsequences.shape}")
        else:
            time_sequences = None

        # N_SAMPLES = len(ordered_list_of3dsequences)
        n_samples, time_wind, time_ftrs = ordered_list_of3dsequences.shape

        if n_samples <= 10:
            raise ValueError(
                    f"Too few samples! {n_samples}, shape:{ordered_list_of3dsequences.shape}")

        if full_eval:
            ses_start = 0
            ses_end = len(ordered_list_of3dsequences) - 1
            # print(f"Full eval: {ses_start}: {ses_end} ({len(ordered_list_of3dsequences)})")
            # ses_end = 5
        else:
            if random_start:
                if n_samples - 1 - session_size < 0:
                    ses_start = 0
                else:
                    ses_start = np.random.randint(0, n_samples - 1 - session_size)
            else:
                "Deterministic start"
                ses_start = 0

            ses_end = ses_start + session_size

            if ses_end > n_samples:
                print(f"Reducing ses end to: {n_samples} from {ses_end}")
                ses_end = n_samples

            # print(f"Partial eval: {ses_start}: {ses_end} ({len(ordered_list_of3dsequences)})")

        print(
                f"Eval sess: {i_eval_sess} selected segment: {segm_i}, "
                f"random: {random_start}, full: {full_eval}, "
                f"start: {ses_start}: end{ses_end}")
        agents_discrete_states, hidden_states = initialize_agents(AGENTS_N)

        "Actions:"
        "0, 1, 2"
        "Buy, Pass, Sell"
        # starttime = time.time()

        how_many_actions = 0
        how_many_valid = 0

        labels = [
                'i_sample',
                'Cash', 'Cargo', 'Action', 'Gain',  # 1,2,3,4
                'Q1', 'Q2', 'Q3', 'price',  # 5,6,7,8
                "timestamp_s"  # 9
        ]
        plot_array = np.zeros((0, len(labels)), dtype=float)

        logged_actions = []

        for i_sample in range(ses_start, ses_end):  # Never in done state

            timesegment_2d = ordered_list_of3dsequences[i_sample, :]
            timesegment_stacked = np.tile(timesegment_2d[np.newaxis, :, :], (AGENTS_N, 1, 1))

            # if not i_sample % WALK_INTERVAL_DEBUG:
            #     print(f"Walking sample: {i_sample}")
            timestamp_s = timesegment_2d[-1, timestamp_col]

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

            cur_step_price = ordered_list_of3dsequences[i_sample, -1, price_col_ind]
            # print(datalist_2dsequences_ordered_train.shape)
            # print(cur_step_price, type(cur_step_price))

            "Dont train"
            new_states, new_hidden_states = resolve_actions_func(
                    cur_step_price, agents_discrete_states, hidden_states, actions
            )
            step_gain = 0.3

            if time_sequences is not None:
                sample_time = time_sequences[i_sample]
                # print(f"Adding action to filesaver: {actions[0]}")
                if actions[0] != 1:
                    logged_actions.append((sample_time, actions[0], cur_step_price))
            else:
                sample_time = None
            plot_vec = [
                    i_sample,
                    hidden_states[0, 0], hidden_states[0, 2], actions[0],
                    step_gain, *q_vals[0, :], cur_step_price,
                    sample_time,
            ]

            plot_vec = np.array(plot_vec).reshape(1, -1)

            # print(plot_array.shape, plot_vec.shape)
            plot_array = np.concatenate([plot_array, plot_vec], axis=0)

            agents_discrete_states = new_states
            hidden_states = new_hidden_states

        # tend_walking = time.time()
        # print(f"End cargo: {hidden_states[0, 2]} and price: {cur_step_price}")
        end_gain = hidden_states[0, 0] - hidden_states[0, 1] + hidden_states[0, 2] * cur_step_price

        plt.subplots(3, 1, figsize=(20, 10), dpi=200, height_ratios=[3, 1, 3])
        # gain = hidden_states[:, 0] - hidden_states[:, 1]
        # x = plot_array[:, 0]
        xtmps = plot_array[:, 9]
        xtmps -= xtmps[0]
        x = xtmps

        plt.subplot(3, 1, 1)
        # for i, lb in enumerate(labels[1:3], 1):
        #     plt.plot(x, plot_array[:, i], label=lb, color=colors[i], alpha=0.8, linewidth=2)
        for act in [0, 1, 2]:
            mask = plot_array[:, 3] == act
            xa = xtmps[mask]
            ya = plot_array[mask, 8]
            lb = {0: "Buy", 1: "Pass", 2: "Sell"}[act]
            s = {0: 35, 1: 15, 2: 40}[act]
            plt.scatter(xa, ya, label=f"Action: {lb}", s=s)

        # print()
        eval_values.append((how_many_actions, how_many_valid, np.round(end_gain, 5)))

        plt.plot(xtmps, plot_array[:, 8], label="Price", color=colors[0], alpha=0.6, linewidth=2)
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
        for i, lb in enumerate(labels[5:8], 5):
            lb = {5: "Q1:Buy", 6: "Q2:Pass", 7: "Q3:Sell"}[i]
            plt.plot(x, plot_array[:, i], label=lb, color=colors[i], alpha=0.8, linewidth=2)

        plt.title("Q vals")
        plt.legend()

        plt.suptitle(naming_ob.path + f", Game: {i_eval_sess}")
        plt.xlabel("sample number")
        plt.tight_layout()
        plt.savefig(os.path.join(path_this_model_folder, "evals", f"eval-{name}-{i_eval_sess}.png"))
        plt.close()
        print(f"Saved fig: {naming_ob.path} - eval - {name} - {i_eval_sess}")

        with open(os.path.join(path_this_model_folder, 'evals', f'eval-{name}-{i_eval_sess}.csv'),
                  "wt")as fp:
            fp.write(f"#Game start: {time_sequences[ses_start]}s\n")
            fp.write("timestamp_s,action,price\n")

            if logged_actions:
                for a, b, c in logged_actions:
                    fp.write(f"{a},{b},{c}\n")

                print(f"Saved actions to: eval-{name}-{i_eval_sess}.csv")
            else:
                print("Not save actions")

    return naming_ob.path, eval_values

    # if allow_train:
    #     model_keras.save_weights(path_this_model_folder + "weights.keras")
    #     RUN_LOGGER.info(f"Saved weights: {naming_ob}")


    # return history, best, best_all


def single_model_evaluate(
        counter, model_params, train_segments, price_id,
        game_n, game_duration,
        name="",
        full_eval=False,
        timestamp_col=None,
):
    "LIMIT GPU BEFORE BUILDING MODEL"

    "GLOBAL LOGGERS"
    # DISCOUNT = 0  # .95
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        (arch_num, time_feats, time_window, float_feats, out_size,
         nodes, lr, batch, loss, discount, iteration,
         ) = model_params
        model = model_builder(
                arch_num,
                time_feats, time_window, float_feats, out_size,
                loss, nodes, lr,
                iteration=iteration,
        )
        reward_fnum = 6

        naming_ob = NamingClass(
                arch_num, iteration=iteration,
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
        result = eval_func(
                model, train_segments,
                price_col_ind=price_id,
                naming_ob=naming_ob,
                session_size=game_duration,
                reward_f_num=reward_fnum,
                game_n=game_n,
                name=name,
                timestamp_col=timestamp_col,
                full_eval=full_eval,
                # time_sequences=time_sequences,
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
    tf.keras.backend.clear_session()
    del train_segments
    print("Cleared memory... ?")

    collected = gc.collect()
    print(f"Collected: {collected}")
    tf.keras.backend.clear_session()


def evaluate_pipeline(
        train_segments, price_col,
        time_wind, time_ftrs,
        float_feats=16, out_size=3,
        workers=4, games_n=5,
        # uniform_games=True,
        game_duration=250, name="",
        # time_sequences=None,
        timestamp_col=None,
        full_eval=False,
):
    # gen1 = grid_models_generator(time_ftrs, time_wind, float_feats=float_feats, out_size=out_size)
    gen_i23 = grid_models_generator_it23(time_ftrs, time_size, float_feats=float_feats,
                                         out_size=output_size)
    # gen_it2 = grid_models_generator_it2(time_ftrs, time_wind, float_feats=float_feats, out_size=out_size)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        process_list = []
        for counter, data in enumerate(gen_i23):
            # if counter not in [1, 3]:
            #     continue
            # if counter != 7:
            #     continue
            proc = executor.submit(
                    single_model_evaluate, *data, train_segments, price_col,
                    games_n, game_duration, name,
                    # time_sequences=time_sequences,
                    full_eval=full_eval,
                    timestamp_col=timestamp_col,
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
        # print("results:")
        # print(results)

        try:
            tab = unpack_evals_to_table(results)
            # print(tab)
            now = datetime.datetime.now()
            dt_str = f"{now.month}.{now.day}-{now.hour}.{now.minute}"
            with open(os.path.join(path_models, f"evals-{name}-{dt_str}.txt"), "wt") as fp:
                fp.write(str(tab))
                fp.write("\n")

            print(f"Saved evals to: evals-{name}-{dt_str}.txt")

        except Exception as err:
            print(err)
            print(f"Results: {results}")


if __name__ == "__main__":
    use('ggplot')

    do_window = False

    files = ["obv_600.txt"]
    # files = ["obv_600.txt", "on_balance_volume.txt"]
    # files = ["on_balance_volume.txt"]
    for file_name in files:
        file_path = path_data_folder + file_name
        if not os.path.isfile(file_path):
            print(f"Skipping file: {file_name}")
            continue

        name, *_ = file_name.split(".")

        if do_window:
            interval_s = 10
            time_size = 50
            float_feats = 1
            output_size = 3

            segments, columns = preprocess_pipe_uniform(
                    file_path, include_time=True,
                    interval_s=interval_s,
                    add_timediff_feature=True,
            )

            column = columns[0]

            timestamp_ind = np.argwhere(column == 'timestamp_s').ravel()[0]  # 0 probably
            price_ind = np.argwhere(column == 'last').ravel()[0] - 1  # offset to dropped time column
            print(f"Time col: {timestamp_ind}, Price col: {price_ind}")

            samples_n, time_ftrs = segments[0].shape
            time_ftrs -= 1  # Reduce due to time column

            trainsegments_ofsequences3d = [to_sequences_forward(segment, time_size, [1])[0] for segment
                                           in segments]

            print("single segment", trainsegments_ofsequences3d[0].shape)
            print(f"Time ftrs: {time_ftrs}, Time window: {time_size}")
        else:
            time_size = 50
            output_size = 3
            # time_ftrs = 0
            float_feats = 1

            trainsegments_ofsequences3d, columns = preprocess_pipe_bars(
                    file_path, get_n_bars=time_size,
                    add_timediff_feature=True,
                    include_timestamp=True,
                    # first_sample_date="2023-6-29",  # only for on_balance_volume
            )
            price_ind = np.argwhere(columns[0] == 'last').ravel()[0]
            timestamp_ind = np.argwhere(columns[0] == 'timestamp_s').ravel()[0]
            print(f"Timestamp: {timestamp_ind}, price: {price_ind}")
            print(columns[0])
            samples_n, _, time_ftrs = trainsegments_ofsequences3d[0].shape
            time_ftrs -= 1  # subtract timestamp

        trainsegments_ofsequences3d = trainsegments_ofsequences3d[:60]

        evaluate_pipeline(
                trainsegments_ofsequences3d, price_ind,
                time_wind=time_size, time_ftrs=time_ftrs,
                float_feats=float_feats, out_size=output_size,
                game_duration=1000,
                workers=3,
                games_n=40,
                name=f"{name}",
                # time_sequences=timestamps_s,
                timestamp_col=timestamp_ind,
                full_eval=False,
        )
        break
