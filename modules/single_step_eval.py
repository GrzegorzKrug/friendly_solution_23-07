import numpy as np

import time
from tensorflow import keras
import os
import datetime

# from random import sample, shuffle
from actors import initialize_agents, resolve_actions_multibuy, resolve_actions_singlebuy

from common_settings import ITERATION, path_data_clean_folder, path_models, path_data_folder
from common_functions import (
    NamingClass, get_splits, get_eps, to_sequences_forward, load_data_split,
    unpack_evals_to_table,
)
# from reward_functions import RewardStore

# from functools import wraps
from collections import deque
from model_creator import grid_models_generator, model_builder

# from io import TextIOWrapper
from yasiu_native.time import measure_real_time_decorator

import traceback
import multiprocessing

import tensorflow as tf

# import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.style import use
import pandas as pd

from preprocess_data import preprocess, generate_interpolated_data
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

    eval_values = []
    if full_eval:
        game_n = len(segments_oftraindata)
        print(f"Full eval, games: {game_n}")

    for i_eval_sess in range(game_n):
        "RANDOM SEGMENT"
        ordered_list_of3dsequences = segments_oftraindata[i_eval_sess]

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
            print(f"Full eval: {ses_start}: {ses_end} ({len(ordered_list_of3dsequences)})")
            ses_end = 5
        else:
            ses_start = np.random.randint(0, n_samples - 1 - session_size)
            ses_end = ses_start + session_size
            print(f"Partial eval: {ses_start}: {ses_end} ({len(ordered_list_of3dsequences)})")

        agents_discrete_states, hidden_states = initialize_agents(agents_n)

        "Actions:"
        "0, 1, 2"
        "Sell, Pass, Buy"
        starttime = time.time()

        how_many_actions = 0
        how_many_valid = 0

        labels = ['i_sample', 'Cash', 'Cargo', 'Action', 'Gain', 'Q1', 'Q2', 'Q3', 'price']
        plot_array = np.zeros((0, len(labels)), dtype=float)

        logged_actions = []

        for i_sample in range(ses_start, ses_end):  # Never in done state

            timesegment_2d = ordered_list_of3dsequences[i_sample, :]
            timesegment_stacked = np.tile(timesegment_2d[np.newaxis, :, :], (agents_n, 1, 1))

            # if not i_sample % WALK_INTERVAL_DEBUG:
            #     print(f"Walking sample: {i_sample}")

            "MOVE TO IF BELOW"

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

            plot_vec = [
                    i_sample,
                    hidden_states[0, 0], hidden_states[0, 2], actions[0],
                    step_gain, *q_vals[0, :], cur_step_price,
            ]

            if time_sequences is not None:
                sample_time = time_sequences[i_sample]
                # print(f"Adding action to filesaver: {actions[0]}")
                if actions[0] != 1:
                    logged_actions.append((sample_time, actions[0], cur_step_price))

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
         nodes, lr, batch, loss, discount
         ) = model_params
        model = model_builder(
                arch_num,
                time_feats, time_window, float_feats, out_size,
                loss, nodes, lr
        )
        reward_fnum = 6

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
    tf.keras.backend.clear_sesion()
    del train_segments
    print("Cleared memory... ?")

    collected = gc.collect()
    print(f"Collected: {collected}")
    tf.keras.backend.clear_sesion()


def run_single_evals_for_allmodels():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    time_wind = 10
    float_feats = 1
    out_sze = 3

    files = ["obv_600.txt"]
    dtnow = datetime.datetime.now()
    datetime_postfix = f"{dtnow.month}.{dtnow.day}-{dtnow.hour}.{dtnow.minute}"

    for file in files:
        file_path = path_data_folder + file
        file_name, *_ = file.split(".")

        if file == "on_balance_volume.txt":
            cut_date = "2023-6-15"
        else:
            cut_date = None
        # print(f"cut date: {cut_date}")

        print(f"Trying: {file_path}")
        if not os.path.isfile(file_path):
            print(f"Skipping file: {file}")
            continue

        gen = grid_models_generator(16, 10, 1, 3)

        interval_s = 10
        split_interval_s = 1800

        # for model in gen:
        # last_state = (0, 0)

        for counter, model_params in gen:
            (arch_num, time_feats, time_window, float_feats, out_size,
             nodes, lr, batch, loss, discount
             ) = model_params
            model_keras = model_builder(
                    arch_num,
                    time_feats, time_window, float_feats, out_size,
                    loss, nodes, lr
            )

            naming_ob = NamingClass(
                    arch_num, iteration=ITERATION,
                    time_feats=time_feats, time_window=time_window, float_feats=float_feats,
                    outsize=out_size,
                    node_size=nodes, reward_fnum=6,
                    learning_rate=lr, loss=loss, batch=batch,
                    discount=discount,
            )
            "GET MODEL PATH"
            path_this_model_folder = os.path.join(path_models, naming_ob.path, "")
            if os.path.isfile(path_this_model_folder + "weights.keras"):
                # RUN_LOGGER.info(f"Loading model: {path_this_model_folder}")
                model_keras.load_weights(path_this_model_folder + "weights.keras")
                print(f"Loaded weights: {naming_ob}")

            else:
                print(f"Not found model for evaluation: {naming_ob.path}")
                continue

            os.makedirs(os.path.join(path_this_model_folder, "single_bar_eval"), exist_ok=True)

            state = 0
            last_segment_end = 0
            was_ok = True
            # last_data_shp = (0, 0)

            "LOADING NEW DATA"
            out_df = pd.read_csv(file_path)
            out_df['action'] = -1

            save_path = os.path.join(path_this_model_folder, "single_bar_eval",
                                     f"{file_name}-{datetime_postfix}.csv")

            with open(save_path, "at") as fp:
                columns = out_df.columns
                ct = ",".join(cl for cl in columns)
                fp.write(f"{ct}\n")

                for i in range(1, len(out_df)):
                    t0 = time.time()
                    loaded_df = pd.read_csv(file_path).iloc[last_segment_end:i + 1, :]

                    "CLEAN"
                    dataframe = preprocess(loaded_df, first_sample_date=cut_date)

                    if len(dataframe) <= 1:
                        print(f"{i} RESET: Skipping iteration: {i}. Df too short: {dataframe.shape}")
                        row = out_df.iloc[i]
                        fp.write(','.join(map(str, row)))
                        fp.write("\n")
                        state = 0

                        if was_ok:
                            last_segment_end = i - 1
                            was_ok = False
                        continue

                    segments, columns = generate_interpolated_data(
                            dataframe=dataframe, include_time=False,
                            interval_s=interval_s, split_interval_s=split_interval_s
                    )

                    # list_ofsequences = [to_sequences_forward(arr, 10, [1])[0] for arr in segments]
                    current_sequence, _ = to_sequences_forward(segments[-1], 10, [1])
                    # print(f"current sequence: {current_sequence.shape}")

                    if len(current_sequence) <= 0:
                        print(f"{i} RESET: Skipping iteration: {i} too short sequence")
                        row = out_df.iloc[i]
                        fp.write(','.join(map(str, row)))
                        fp.write("\n")
                        state = 0
                        if was_ok:
                            last_segment_end = i - 1
                            was_ok = False
                        continue

                    pred_state = np.array(state).reshape(1, 1)
                    pred_arr = current_sequence[-1, :, :].reshape(1, time_wind, 16)
                    predicted = model_keras.predict([pred_arr, pred_state], verbose=False)
                    act = np.argmax(predicted, axis=1)[0]

                    out_df.loc[i, 'action'] = act
                    row = out_df.iloc[i]
                    fp.write(','.join(map(str, row)))
                    fp.write("\n")

                    "Post state eval"
                    if act == 0:
                        state = 1
                    elif act == 2:
                        state = 0

                    loop_dur = time.time() - t0
                    print(i,
                          f"Loop duration: {loop_dur:>5.2}s",
                          f"Act: {act}, {i / len(out_df) * 100:>4.1f}% (index {last_segment_end}:{i + 1})")
                    was_ok = True

            del model_keras
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()


def start_continous_eval(
        input_file_path, output_file_path, model,
        start_on_last_sample=True,
        predict_each_new_bar=True,
):
    pass


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import watchdog


# class FileHandler(FileSystemEventHandler):
#     def __init__(self, callback):
#         # self.filename = filename
#         self.callback = callback
#         # self.last_line = None
#
#     def on_any_event(self, event):
#         print(f"Any event: {event}")
#
#     def on_modified(self, event):
#         print("Modified event")
#         self.callback()
class FilePatternHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        watchdog.events.PatternMatchingEventHandler.__init__(
                self,
                patterns=["*.txt"],
                ignore_directories=True,
                case_sensitive=False,
        )

    def on_any_event(self, event):
        print(f"Any event occured: {event}")


def dummy(*a):
    print(f"A: {a}")


def start_stream_predict(
        input_filepath,

        time_feats=16,
        time_window=10,
        float_feats=1,
        out_size=3,

        arch_num=101,
        iteration=0,
        nodes=1000,
        reward_fnum=6,
        lr="1e-05",
        loss='mae',
        batch=500,
        discount='0.9',

        split_interval_s=1800,
        interval_s=10,
        output_filepath=None,
):
    model_keras = model_builder(
            arch_num,
            time_feats, time_window, float_feats, out_size, loss, nodes, lr,
            iteration=iteration,
    )
    naming_ob = NamingClass(
            arch_num, iteration=ITERATION,
            time_feats=time_feats, time_window=time_window, float_feats=float_feats,
            outsize=out_size,
            node_size=nodes, reward_fnum=reward_fnum,
            learning_rate=lr, loss=loss, batch=batch,
            discount=discount,
    )

    weights_path = os.path.join(path_models, naming_ob.path, "weights.keras")
    if os.path.isfile(weights_path):
        print("Got weights.")
        model_keras.load_weights(weights_path)
    else:
        print(f"Found no weights: {weights_path}")
        raise ValueError(f"Model not found: {weights_path}")

    state = 0
    last_segment_end = 0
    last_size = os.path.getsize(input_filepath)
    loadead_df = pd.read_csv(input_filepath)
    last_bar_ind = len(loadead_df) - 1
    was_ok = True
    if output_filepath is None:
        output_filepath = os.path.join(
                os.path.dirname(input_filepath),
                f"test_{arch_num:>03}_{lr}_{loss}.txt"
        )

    was_file = os.path.isfile(output_filepath)

    print("READY FOR NEW SAMPLES")
    print(f"Model: {naming_ob.path}")
    print(f"Output file: {output_filepath}")

    with open(output_filepath, "at", buffering=1) as fp:
        if not was_file:
            columns = loadead_df.columns
            ct = ",".join(cl for cl in columns)
            fp.write(f"{ct}\n")

        while True:
            time.sleep(0.1)
            size = os.path.getsize(input_filepath)
            if last_size == size:
                continue

            last_size = size
            # print(f"New size: {size}")

            loadead_df = pd.read_csv(input_filepath)
            # print(f"Shape: {loadead_df.shape}")
            missing_predictions = len(loadead_df) - 1 - last_bar_ind
            print(f"File has changed by {missing_predictions} entries")
            # print(f"Missing predictions: {missing_predictions}")

            for i in range(missing_predictions):
                # dataframe = loadead_df.iloc[last_segment_end:last_bar_ind + i + 2]
                dataframe = loadead_df.iloc[last_segment_end:last_bar_ind + i + 2].copy()
                # print(f"Predicting from: {last_segment_end}:{last_bar_ind + i + 2}")
                t0 = time.time()

                "CLEAN"
                dataframe = preprocess(dataframe, first_sample_date=None)
                if len(dataframe) <= 1:
                    print(
                            f"{last_bar_ind + i + 1} RESET: Skipping iteration: {i}. Df too short: {dataframe.shape}")
                    # row = out_df.iloc[i]
                    ser = loadead_df.iloc[last_bar_ind + 1 + i]
                    fp.write(','.join(map(str, ser)))
                    fp.write(",-1")
                    fp.write("\n")
                    state = 0

                    if was_ok:
                        last_segment_end = last_bar_ind + 1 + i
                        was_ok = False
                    continue

                segments, columns = generate_interpolated_data(
                        dataframe=dataframe, include_time=False,
                        interval_s=interval_s, split_interval_s=split_interval_s
                )
                # print(f"Splitted into {len(segments)} segments")

                # list_ofsequences = [to_sequences_forward(arr, 10, [1])[0] for arr in segments]
                current_sequence, _ = to_sequences_forward(segments[-1], 10, [1])
                # print(f"current sequence: {current_sequence.shape}")

                if len(current_sequence) <= 0:
                    print(f"{last_bar_ind + i + 1} RESET: Skipping iteration: {i} too short sequence")
                    ser = loadead_df.iloc[last_bar_ind + 1 + i]
                    # ser['act'] = -1
                    fp.write(','.join(map(str, ser)))
                    fp.write(",-1")
                    fp.write("\n")
                    state = 0
                    if was_ok:
                        last_segment_end = last_bar_ind + 1 + i
                        was_ok = False
                    continue

                # print(f"Predicting from sequences: {current_sequence.shape}")

                pred_state = np.array(state).reshape(1, 1, 1)
                # pred_arr = current_sequence[-1, :, :][np.newaxis, :, :]
                pred_arr = current_sequence[-1, :, :].reshape(1, time_window, 16)
                # print(f"Pred state: {pred_state}, {pred_state.shape}")
                # print(f"Predict shapes: {pred_arr.shape}, {pred_state.shape}")
                predicted = model_keras.predict([pred_arr, pred_state], verbose=False)
                act = np.argmax(predicted, axis=1)[0]
                print(f"Predicted: {predicted}")
                # print(f"Act: {act} from state: {state}")

                ser = loadead_df.iloc[last_bar_ind + 1 + i]
                fp.write(','.join(map(str, ser)))
                fp.write(f",{act}")
                fp.write("\n")

                prev_state = state
                "Post state eval"
                if act == 0:
                    state = 1

                elif act == 2:
                    state = 0

                loop_dur = time.time() - t0
                print(last_bar_ind + i + 1, i,
                      f"Loop duration: {loop_dur:>5.2}s",
                      f"Act: {act}, End state:{state}, was state: {prev_state}, (index {last_segment_end}:{last_bar_ind + 2 + i})")
                was_ok = True

            last_bar_ind = len(loadead_df) - 1
            print("========")


if __name__ == "__main__":
    # input_filepath = os.path.join(path_data_folder, "test_updating.txt")
    input_filepath = os.path.join("mnt", "export", "obv_600.txt")
    print(f"Is file: {os.path.isfile(input_filepath)}, {input_filepath}")

    # start_stream_predict(input_filepath, arch_num=1, loss='huber', lr='1e-06', )
    # start_stream_predict(input_filepath, arch_num=103, loss='mae', lr='1e-06', )
    # start_stream_predict(input_filepath, arch_num=101, loss='mae', lr='1e-05', )
    # start_stream_predict(input_filepath, arch_num=1, loss='mae', lr='1e-06', )
