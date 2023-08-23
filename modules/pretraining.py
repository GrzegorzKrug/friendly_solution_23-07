import numpy as np

import time
from tensorflow import keras
import datetime
import logger

from random import sample, shuffle
from actors import initialize_agents, resolve_actions_multibuy, resolve_actions_singlebuy

from common_settings import path_data_clean_folder, path_models, path_data_folder
from common_functions import NamingClass, get_splits, get_eps, to_sequences_forward, load_data_split
from reward_functions import RewardStore

from preprocess_data import preprocess_pipe_uniform, preprocess_pipe_bars
from functools import wraps
from collections import deque
from model_creator import (
    grid_models_generator, grid_models_generator_2,
    grid_models_generator_it23,
    model_builder,
)

from io import TextIOWrapper
from yasiu_native.time import measure_real_time_decorator

# import traceback
# import multiprocessing as mpc

import tensorflow as tf

import gc
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import pickle


# session_dataframe.loc[
#     len(session_dataframe)] = session_eps, i_train_sess, ses_start, ses_end, g


class AgentsMemory:
    def __init__(self, ):
        # self.end_inds = []
        # self.end_inds_fut = []
        # self.states = []
        # self.states_fut = []
        # self.rewards = []
        # self.actions = []
        self.memory = []

    def add_sample(
            self, env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, reward, done,
            qvals
    ):
        self.memory.append((
                env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, reward, done,
                qvals
        ))
        # for i, (ind, inp, fut) in enumerate(zip(timeser_index, agent_states, future_inputs)):
        #     self.env_state[i].append(ind)
        #     self.state_inputs[i].append(inp)
        #     self.future_inputs[i].append(fut)
        #
        # self.actions.append(act)
        # self.rewards.append(rew)
        # self.qvals.append(qvals)

    def add_batch(self, *args):
        for batch in zip(*args):
            # print(f"Adding batch: {batch}")
            self.add_sample(*batch)

    def random_samples(self, fraction=0.8):
        n = int(len(self.memory) * fraction)
        # print(f"Random N:{n} samples")
        return sample(self.memory, n)

    def __str__(self):
        return f"AgentsMemory(Size 6 lists each: {len(self.memory)})"

    def __len__(self):
        return len(self.memory)


def results_decorator(fun):
    @wraps(fun)
    def wrapper(*arg, naming_ob=None, **kw):
        # print(f"Args:{arg}")
        # print(f"Naming: {naming_ob}")
        # print(f"Kw: {kw}")
        cur_date = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M")

        path_this_model_folder = os.path.join(path_models, naming_ob.path, "")

        os.makedirs(path_this_model_folder, exist_ok=True)
        os.makedirs(os.path.join(path_this_model_folder + "data"), exist_ok=True)

        path_qvals = os.path.join(path_models, naming_ob.path, 'data', f'{cur_date}-qvals.csv')
        path_sess = os.path.join(path_models, naming_ob.path, 'data', f'{cur_date}-sess.csv')
        path_times = os.path.join(path_models, naming_ob.path, 'data', f'{cur_date}-times.csv')
        path_loss = os.path.join(path_models, naming_ob.path, 'data', f'{cur_date}-loss.csv')
        path_rewards = os.path.join(path_models, naming_ob.path, 'data', f'{cur_date}-rewards.csv')

        was_qvals = os.path.isfile(path_qvals)
        was_sess = os.path.isfile(path_sess)
        was_times = os.path.isfile(path_times)
        was_loss = os.path.isfile(path_loss)
        was_rewards = os.path.isfile(path_rewards)

        with open(path_qvals, "at", buffering=1) as q_textfile:
            if not was_qvals:
                arr = ['sess_eps', 'i_train_sess', 'agent_i', 'i_sample', 'q1', 'q2', 'q3']
                q_textfile.write(','.join(arr) + "\n")

            with open(path_sess, "at", buffering=1) as sess_textfile:
                if not was_sess:
                    arr2 = ['sess_eps', 'i_train_sess', 'sess_start', 'sess_end', 'agent_i', 'gain']
                    sess_textfile.write(','.join(arr2) + "\n")

                with open(path_times, "at", buffering=1) as time_textfile:
                    if not was_times:
                        time_textfile.write("loop_time\n")

                    with open(path_loss, "at", buffering=1) as loss_textfile:
                        if not was_loss:
                            loss_textfile.write("sess_i,fresh_loss,oldmem_loss\n")

                        with open(path_rewards, "at", buffering=1) as rew_textfile:
                            if not was_rewards:
                                rew_textfile.write("sess_i,eps,i_sample,agent_i,reward,\n")

                            return fun(
                                    *arg,
                                    naming_ob=naming_ob, **kw,
                                    qvals_file=q_textfile, session_file=sess_textfile,
                                    time_file=time_textfile, loss_file=loss_textfile,
                                    rew_file=rew_textfile,
                            )

    return wrapper


# @results_decorator
def pretrain_qmodel(
        model_keras: keras.Model, segmentslist_alldata3d,
        price_col_ind,
        naming_ob: NamingClass,
        agents_n=20,
        games_n=1000,
        game_duration=3600,

        # model_memory: ModelMemory = None,

        # Optional
        max_eps=0.5, override_eps=None,
        remember_fresh_fraction=0.2,
        train_from_oldmem_fraction=0.4,
        epochs=20,
        batch_train=15000,

):
    RUN_LOGGER.debug(f"Train params: {naming_ob}: trainN:{games_n}, agents: {agents_n}.")
    # WALK_INTERVAL_DEBUG = 250
    RUN_LOGGER.debug(f"Input samples: {sum(map(len, segmentslist_alldata3d))}")

    # N_SAMPLES = len(datalist_2dsequences_ordered_train)

    "GET MODEL PATH"
    path_this_model_folder = os.path.join(path_models, naming_ob.path, "")
    if os.path.isfile(path_this_model_folder + "weights.keras"):
        RUN_LOGGER.info(f"Loading model: {path_this_model_folder}")
        model_keras.load_weights(path_this_model_folder + "weights.keras")
    else:
        RUN_LOGGER.warning(f"Not loading model - {naming_ob.path}.")

    time_memory, float_memory, action_memory = [], [], []
    action_price_cost = 0.01
    minibatch_size = int(naming_ob.batch)

    for segment_i, segment in enumerate(segmentslist_alldata3d):
        # print()
        # print(f"Segment i: {segment_i}")
        if segment_i >= games_n:
            RUN_LOGGER.info(f"Breaking Pretrain loop at: {segment_i}")
            break

        for sample_i in range(len(segment) - 1):  # NEED FUT Sample
            if sample_i >= game_duration:
                RUN_LOGGER.info(
                        f"Breaking loop at: {segment_i} - {sample_i} ({sample_i}>={game_duration}) (size: {len(segment)})")
                break

            state = segment[sample_i]
            price_samp = segment[sample_i, -1, price_col_ind]
            price_fut = segment[sample_i + 1, -1, price_col_ind]
            price_change = np.clip((price_fut * 10000.0 - price_samp * 10000.0), -5, 5) * 100
            # print(f"PRICE CHANGE: {price_change:>5.5f}: {price_samp:>5.7f}, {price_fut:>5.7f}")

            for hid_stat in [0, 1]:
                if hid_stat == 0:
                    "NO CARGO"
                    qvs = [-price_samp / 3 - action_price_cost, -price_change, -10]
                else:
                    "CARGO"
                    qvs = [-10, price_change, price_samp / 2 - action_price_cost]

                time_memory.append(state)
                float_memory.append(hid_stat)
                action_memory.append(qvs)

            if len(float_memory) >= batch_train:
                time_memory = np.array(time_memory)
                float_memory = np.array(float_memory).reshape(-1, 1)
                action_memory = np.array(action_memory)

                time0_pretrain = time.time()
                history_ob = model_keras.fit(
                        [time_memory, float_memory], action_memory,
                        shuffle=True,
                        epochs=epochs,
                        batch_size=minibatch_size,
                        verbose=True
                )

                time_pretrain = time.time() - time0_pretrain
                RUN_LOGGER.debug(
                        f"Pretrained loop: {len(time_memory)}, took: {time_pretrain:>4.2f}s, Loss: {history_ob.history['loss']}")

                time_memory = []
                float_memory = []
                action_memory = []

    if len(float_memory) >= minibatch_size:
        time_memory = np.array(time_memory)
        float_memory = np.array(float_memory).reshape(-1, 1)
        action_memory = np.array(action_memory)

        time0_pretrain = time.time()
        history_ob = model_keras.fit(
                [time_memory, float_memory], action_memory,
                shuffle=True,
                epochs=epochs,
                batch_size=minibatch_size,
                verbose=False)

        time_pretrain = time.time() - time0_pretrain
        RUN_LOGGER.debug(
                f"Pretrained-LastTrain: {len(time_memory)}, took: {time_pretrain:>4.2f}s, Loss: {history_ob.history['loss']}")

        time_memory = []
        float_memory = []
        action_memory = []

    model_keras.save_weights(path_this_model_folder + "weights.keras")
    RUN_LOGGER.info(f"PRETRAIN Saved weights: {naming_ob}")

    ret = gc.collect()
    print(f"Loop collected: {ret}")
    tf.keras.backend.clear_session()
    ret = gc.collect()
    print(f"Loop collected: {ret}")


def save_csv_locked(df, path):
    while True:
        try:
            df.to_csv(path)
            break
        except PermissionError as er:
            RUN_LOGGER.warn(f"Waiting for file access ({path}): {er}")
            time.sleep(2)


def single_model_pretraining_function(
        counter, model_params, train_sequences, price_ind,
        games_n, game_duration,
        main_logger: logger,
        override_params=dict(),
):
    "LIMIT GPU BEFORE BUILDING MODEL"
    main_logger.info(f"Process of: {counter} has started now.")

    "GLOBAL LOGGERS"
    global RUN_LOGGER
    global DEBUG_LOGGER
    RUN_LOGGER = logger.create_logger(number=counter)
    DEBUG_LOGGER = logger.create_debug_logger(number=counter)
    # DISCOUNT = 0  # .95

    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        (arch_num, time_feats, time_window, float_feats, out_size,
         nodes, lr, batch, loss, discount, iteration
         ) = model_params

        model = model_builder(
                arch_num, time_feats, time_window, float_feats, out_size,
                loss, nodes, lr, iteration=iteration, override_params=override_params,
        )
        reward_fnum = 6

        RUN_LOGGER.info(
                f"Starting {counter}: Arch Num:{arch_num} Version:? Loss:{loss} Nodes:{nodes} Batch:{batch} Lr:{lr}")
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
        RUN_LOGGER.error(exc, exc_info=True)
        main_logger.error(exc, exc_info=True)
        return None

    try:
        # for gpu in tf.config.experimental.list_physical_devices("GPU"):
        #     pass
        # f"Limitig gpu: {gpu}"
        # tf.config.experimental.set_memory_growth(gpu, True)

        path_this_model_folder = os.path.join(path_models, naming_ob.path, "")

        os.makedirs(path_this_model_folder, exist_ok=True)
        os.makedirs(os.path.join(path_this_model_folder + "data"), exist_ok=True)

        pretrain_qmodel(
                model, train_sequences,
                price_col_ind=price_ind,
                naming_ob=naming_ob,
                game_duration=game_duration,
                games_n=games_n,
                # reward_f_num=reward_fnum,
                # discount=discount,
        )
    except Exception as exc:
        "PRINT TO SYS"
        print(f"EXCEPTION: {exc}")
        RUN_LOGGER.error(exc, exc_info=True)

    "Clear memory?"

    "Clear memory?"
    del model
    tf.keras.backend.clear_session()
    del train_sequences
    print("Cleared memory... ?")

    collected = gc.collect()
    print(f"Collected: {collected}")
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    MainLogger = logger.create_logger(name="MainProcess")
    # DEBUG_LOGGER = logger.debug_logger
    MainLogger.info("=== NEW TRAINING ===")

    # interval_s = 10

    include_time = False
    # segments, columns = preprocess_pipe_uniform(
    #         file_path, include_time=include_time, interval_s=interval_s,
    #         add_timediff_feature=True,
    # )
    # print(f"Pipe output, segments:{segments[0].shape}, columns:{columns[0].shape}")

    # train_data = sequences[0]
    # column = columns[0]

    # time_col = column.index('timestamp_s')

    # trainsegments_ofsequences3d = [to_sequences_forward(segment, time_wind, [1])[0] for segment in
    #                                segments]
    # train_sequences, _ = to_sequences_forward(train_data, time_wind, [1])


    time_size = 50
    output_size = 3
    # time_ftrs = 0
    float_feats = 1

    file_path = os.path.join(path_data_folder, "obv_600.txt")
    # file_path = os.path.join(path_data_folder, "on_balance_volume.txt")
    trainsegments_ofsequences3d, columns = preprocess_pipe_bars(
            file_path, get_n_bars=time_size,
            workers=8,
            normalize_timediff=True,
            minsamples_insegment=300,
            # clip_df_left=5000,
            # clip_df_right=4000,
            # first_sample_date="2023-6-29",  # only for on_balance_volume
    )
    price_ind = np.argwhere(columns[0] == 'last').ravel()[0]
    samples_n, _, time_ftrs = trainsegments_ofsequences3d[0].shape

    print(
            f"Segments: {len(trainsegments_ofsequences3d)}, {trainsegments_ofsequences3d[0].shape}, time ftrs: {time_ftrs}")
    print(f"Columns: {columns[0]}")
    print(f"All samples 2d: {sum(map(len, trainsegments_ofsequences3d))}")
    assert time_ftrs == len(columns[0]), \
        f"Columns must mast time_features but go: ftr:{float_feats} and cols: {len(columns[0])}"

    if include_time:
        time_ftrs -= 1

    # float_feats += 1  # add State

    # sys.exit()
    "Model Grid"
    gen_it23 = grid_models_generator_it23(time_ftrs, time_size, float_feats=float_feats,
                                          out_size=output_size)
    trainsegments_ofsequences3d = trainsegments_ofsequences3d[:60]

    print("STARTING LOOP")
    with ProcessPoolExecutor(max_workers=4) as executor:
        process_list = []
        for counter, data in enumerate(gen_it23):
            MainLogger.info(f"Adding process with: {data}")
            games_n = 200
            game_duration = 5000
            # if counter >= 1:
            #     break
            # elif counter <= 11:
            #     train_duration = 280
            # else:
            #     continue

            proc = executor.submit(
                    single_model_pretraining_function, *data, trainsegments_ofsequences3d, price_ind,
                    games_n, game_duration,
                    MainLogger,
                    override_params=dict(lr=1e-4, batch_size=150),
            )
            process_list.append(proc)
            print(f"Added process: {counter}")

        while True:
            "Deleting loop"

            to_del = set()

            for proc in process_list:
                if proc.done():
                    to_del.add(proc)

            for val in to_del:
                MainLogger.info(f"Removing finished process: {val}")
                process_list.remove(val)
                print(val.exception())
                # print(dir(val))

            if len(process_list) <= 0:
                break

            time.sleep(10)
            ret = gc.collect()
            print(f"Loop collected: {ret}")
            tf.keras.backend.clear_session()

    print("Script end....")
