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

import traceback
import multiprocessing as mpc

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


class ModelMemory:
    def __init__(self, agents_mem: AgentsMemory = None, maxlen=200000):
        self.memory = deque(maxlen=maxlen)

        if agents_mem:
            self.migrate(agents_mem.memory)

    def add_sample(
            self, segment_i, env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, reward,
            done,
    ):
        self.memory.append((
                segment_i, env_state_ind, env_state_ind_fut,
                agent_state, agent_state_fut, act, reward, done,
        ))

    @measure_real_time_decorator
    def migrate(self, segment_i, mem_iterable):
        for batch in mem_iterable:
            # print(f"Migrating batch: {batch}")
            self.add_sample(segment_i, *batch[:-1])  # Drop q-vals at end

    # def add_batch(self, *args):
    #     for batch in zip(*args):
    #         # print(f"Adding batch: {batch}")
    #         self.add_sample(*batch)

    def __str__(self):
        return f"ModelMemory (Size 6 lists each: {len(self.memory)})"

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


@results_decorator
def train_qmodel(
        model_keras: keras.Model, segmentslist_alldata3d,
        price_col_ind,
        naming_ob: NamingClass,
        agents_n=20,
        games_n=1000,
        game_duration=3600,

        allow_train=True, model_memory: ModelMemory = None,

        # Optional
        max_eps=0.5, override_eps=None,
        remember_fresh_fraction=0.2,
        train_from_oldmem_fraction=0.4,
        old_memory_size=70_000,
        # refresh_n_times=3,
        # local_minima=None, local_maxima=None,
        # local_minima_soft=None, local_maxima_soft=None,
        reward_f_num=3,
        discount=0.9,
        # director_loc=None, name=None, timeout=None,
        # save_qval_dist=False,
        extra_training_from_oldmemory=3,

        # PARAMS ==============
        # time_window_size=10,
        # stock_price_multiplier=0.5,
        # stock_ammount_in_bool=True,
        # reward_fun: reward_fun_template,
        allow_multibuy=False,
        use_random_segment=True,

        # FILE SAVERS
        qvals_file: TextIOWrapper = None,
        session_file: TextIOWrapper = None,
        time_file: TextIOWrapper = None,
        loss_file: TextIOWrapper = None,
        rew_file: TextIOWrapper = None,
):
    RUN_LOGGER.debug(
            f"Train params: {naming_ob}: trainN:{games_n}, agents: {agents_n}. Reward F:{reward_f_num}")
    WALK_INTERVAL_DEBUG = 250
    RUN_LOGGER.debug(f"Input samples: {sum(map(len, segmentslist_alldata3d))}")

    # N_SAMPLES = len(datalist_2dsequences_ordered_train)

    "GET MODEL PATH"
    path_this_model_folder = os.path.join(path_models, naming_ob.path, "")
    if os.path.isfile(path_this_model_folder + "weights.keras"):
        RUN_LOGGER.info(f"Loading model: {path_this_model_folder}")
        model_keras.load_weights(path_this_model_folder + "weights.keras")
    else:
        RUN_LOGGER.warning(f"Not loading model - {naming_ob.path}.")

    if model_memory is None:
        model_memory = ModelMemory(maxlen=int(old_memory_size))
    else:
        assert isinstance(model_memory, (ModelMemory,)), "Memory must be an instance of ModelMemory"

    if not allow_train:
        agents_n = 1
    if allow_multibuy:
        resolve_actions_func = resolve_actions_multibuy
    else:
        resolve_actions_func = resolve_actions_singlebuy

    reward_fun = RewardStore.get(reward_f_num)

    out_space = int(naming_ob.outsize)
    # qv_dataframe = pd.DataFrame(columns=['eps', 'sess_i', 'sample_n', 'buy', 'idle', 'sell'])
    # session_dataframe = pd.DataFrame(columns=['eps', 'session_num', 'ind_start', 'ind_end', 'gain'])

    LOOP_TIMES = deque(maxlen=100)

    for i_train_sess in range(games_n):
        if use_random_segment:
            segm_i = np.random.randint(0, len(segmentslist_alldata3d))
            session_sequences3d = segmentslist_alldata3d[segm_i]
        else:
            segm_i = 0
            session_sequences3d = segmentslist_alldata3d[segm_i]

        n_samples, time_wind, time_ftrs = session_sequences3d.shape

        last_start = n_samples - game_duration
        # print(f"Last start: {last_start} for: {N_SAMPLES} of size: {session_size}")
        if last_start < 0:
            ses_start = 0
        else:
            ses_start = np.random.randint(0, last_start)

        ses_end = ses_start + game_duration
        if ses_end > n_samples:
            ses_end = n_samples

        starttime = time.time()
        fresh_memory = AgentsMemory()

        if override_eps is not None:
            session_eps = override_eps
        else:
            session_eps = get_eps(i_train_sess, games_n, max_explore=max_eps)

        RUN_LOGGER.info(
                f"Staring session: {i_train_sess + 1} of {games_n} (Eps: {session_eps:>2.3f})@ segm:{segm_i}: Indexes: {ses_start, ses_end}, Segm size: {len(session_sequences3d)}, {naming_ob.path}.")
        # print(f"Data shape: {session_sequences3d.shape}")

        "Start with different money values"
        # "CASH | CARGO | LAST SELL | LAST BUY | LAST TRANSACTION PRICE"

        if allow_train:
            # agents_stocks, current_cash_arr, starting_money_arr = initialize_agents(START, agents_n)
            start_empty_cargo = True
            if start_empty_cargo:
                "Start with just cash"
                agents_discrete_states, hidden_states = initialize_agents(agents_n)
            else:
                "Star with cargo"
                agents_discrete_states, hidden_states = initialize_agents(agents_n, 1)

        else:
            agents_discrete_states, hidden_states = initialize_agents(1)
            # "Validating = 1 Agent"
            # starting_money_arr = np.array([2])
            # agents_stocks = np.array(START, dtype=float).reshape(1, -1)
        # print(f"Initial hidden state: {hidden_states[:, 2]}")

        "Actions:"
        "0, 1, 2"
        "Sell, Pass, Buy"
        t0_walking = time.time()
        session_size = ses_end - ses_start
        for i_sample in range(ses_start, ses_end - 1):  # Never in done state
            done_session = i_sample == (n_samples - 1)  # is this last sample?

            env_arr2d = session_sequences3d[i_sample, :, :]
            env_arr3d = np.tile(env_arr2d[np.newaxis, :, :], (agents_n, 1, 1))

            if not i_sample % WALK_INTERVAL_DEBUG:
                RUN_LOGGER.debug(f"Walking sample: {i_sample}, memory: {len(fresh_memory.memory)}")

            # if done_session:
            #     future_segment3d = None
            # else:
            #     future_segment3d = datalist_2dsequences_ordered_train[i_sample + 1, :][np.newaxis, :, :]
            #   # futuresegment_stacked = np.tile(future_segment3d, (agents_n, 1, 1))


            q_vals = model_keras.predict(
                    [env_arr3d, agents_discrete_states],
                    verbose=False
            )
            "Select Action"
            if allow_train and session_eps > 0 and session_eps > np.random.random():
                "Random Action"
                actions = np.random.randint(0, out_space, agents_n)
                # print(f"Random actions: {actions}")
            # elif FORCE_RANDOM:
            #     actions = np.random.randint(0, 3, agents_n)
            else:
                actions = np.argmax(q_vals, axis=-1)

            "Saving predicted qvals (Save All)"
            for agent_i, qv in enumerate(q_vals):
                # qv_dataframe.loc[len(qv_dataframe)] = session_eps, i_train_sess, i_sample, *qv
                arr = [session_eps, i_train_sess, agent_i, i_sample, *qv]
                text = ','.join([str(a) for a in arr]) + "\n"
                qvals_file.write(text)

            # last_price = prices[x]
            rewards = []
            valids = []

            # env_state_arr = timesegment_2d
            # price_ind
            # price_ind

            cur_step_price = env_arr2d[-1, price_col_ind]
            for agent_i, (state_arr, act, hidden_arr) in enumerate(
                    zip(agents_discrete_states, actions, hidden_states)
            ):
                # hidden_arr[1] = cur_step_price

                rew, valid = reward_fun(
                        env_arr2d, state_arr, act, hidden_arr, done_session, cur_step_price,
                        price_col_ind
                )
                arr = [i_train_sess, session_eps, i_sample, agent_i, rew]
                text = ",".join([str(a) for a in arr])
                rew_file.write(text + "\n")
                rewards.append(rew)
                valids.append(valid)

            if allow_train:
                env_states_inds = [i_sample] * agents_n
                env_states_inds_fut = [i_sample + 1] * agents_n
                # agents_states = agents_discrete_states.copy()
                new_states, new_hidden_states = resolve_actions_func(
                        cur_step_price, agents_discrete_states, hidden_states, actions
                )

                dones = [done_session] * agents_n

                fresh_memory.add_batch(
                        env_states_inds, env_states_inds_fut, agents_discrete_states,
                        new_states,
                        actions, rewards, dones, q_vals
                )

            else:
                "Dont train"
                new_states, new_hidden_states = resolve_actions_func(
                        cur_step_price, agents_discrete_states, hidden_states, actions
                )

            agents_discrete_states = new_states
            hidden_states = new_hidden_states

        tend_walking = time.time()
        RUN_LOGGER.info(
                f"Walking through data took: {tend_walking - t0_walking:>5.4f}s. {(tend_walking - t0_walking) / session_size:>5.5f}s per step")

        gain = hidden_states[:, 0] - hidden_states[:, 1]
        last_price = env_arr2d[-1, price_col_ind]
        gain += hidden_states[:, 2] * last_price
        # gain *= 0
        # RUN_LOGGER.debug(f"End gains: {gain}")

        "Saving session data"
        for gi, g in enumerate(gain):
            arr = [session_eps, i_train_sess, ses_start, ses_end, gi, g]
            text = ','.join([str(a) for a in arr]) + "\n"
            # print(f"Writing text to session: '{text}'")
            session_file.write(text)

        DEBUG_LOGGER.debug(f"End hidden state: {hidden_states}")
        DEBUG_LOGGER.debug(f"End gains: {gain.reshape(-1, 1)}")
        # DEBUG_LOGGER.debug(f"End cargo: {hidden_states[:, 2].reshape(-1, 1)}")
        # RUN_LOGGER.debug(f"End cargo: {hidden_states[:, 2]}")

        "Session Training"
        if allow_train:
            fresh_loss = deep_q_reinforce_fresh(
                    model_keras, fresh_memory.memory,
                    discount=discount,
                    env_alldata3d=session_sequences3d,
                    mini_batchsize=int(naming_ob.batch),
            )
            # L(history.history['loss'])
            # loss_file.write(f"{i_train_sess},{fresh_loss}\n")

            # loss_file.write(f"{i_train_sess},{fresh_loss},{0}\n")

            if len(model_memory) <= (old_memory_size // 2):
                "Migrate full"
                shuffle(fresh_memory.memory)
                model_memory.migrate(segm_i, fresh_memory.memory)

            else:
                "Migrate fraction memory"
                k = int(remember_fresh_fraction * len(fresh_memory))
                model_memory.migrate(segm_i, sample(fresh_memory.memory, k))

            "Train from old"
            k = int(train_from_oldmem_fraction * len(model_memory))
            train_number = max(0, int(extra_training_from_oldmemory)) + 1
            if k > 3000:
                # print(f"Retraining for: {train_number + 1}")
                for tri_i in range(train_number):
                    "Pick random samples"
                    old_samples = sample(model_memory.memory, k)
                    old_loss = deep_q_reinforce_oldmem(
                            model_keras, old_samples,
                            discount=discount,
                            # env_alldata3d=session_sequences3d,
                            segments_of3ddata=segmentslist_alldata3d,
                            mini_batchsize=int(naming_ob.batch),
                    )
                    loss_file.write(f"{i_train_sess},{fresh_loss},{old_loss}\n")
            else:
                for tri_i in range(train_number):
                    loss_file.write(f"{i_train_sess},{fresh_loss},-1\n")

        "RESOLVE END SCORE"

        endtime = time.time()
        duration = endtime - starttime
        LOOP_TIMES.append(duration)
        loop_sum_text = f"This loop took: {duration:>5.4f}s. Fresh mem: {len(fresh_memory)}, Total mem: {len(model_memory)}. Mean loop time: {np.mean(LOOP_TIMES):>4.2f}"
        DEBUG_LOGGER.info(loop_sum_text)
        RUN_LOGGER.info(loop_sum_text)

        if allow_train and not i_train_sess % 10:
            model_keras.save_weights(path_this_model_folder + "weights.keras")
            RUN_LOGGER.info(f"Saved weights: {naming_ob}")

        time_file.write(f"{duration}\n")
        # DEBUG_LOGGER.debug(f"Mean loop time = {np.mean(LOOP_TIMES) / 60:4.4f} m")
        # RUN_LOGGER.info(f"Mean loop time = {np.mean(LOOP_TIMES) / 60:4.4f} m")

    if allow_train:
        model_keras.save_weights(path_this_model_folder + "weights.keras")
        RUN_LOGGER.info(f"Saved weights: {naming_ob}")

    # ref = sys.getrefcount(model_memory)
    # print(f"Memory references: {ref}")
    # ref = sys.getrefcount(model_memory.memory)
    # print(f"Memory.memory references: {ref}")

    # ref = sys.getrefcount(fresh_memory)
    # print(f"Fresh Memory references: {ref}")
    # ref = sys.getrefcount(fresh_memory.memory)
    # print(f"Fresh Memory.memory references: {ref}")

    model_memory.memory = []
    fresh_memory.memory = []

    del model_memory
    del fresh_memory

    # ref = sys.getrefcount(model_memory)
    # print(f"Memory references: {ref}")
    # ref = sys.getrefcount(model_memory.memory)
    # print(f"Memory.memory references: {ref}")
    #
    # ref = sys.getrefcount(fresh_memory)
    # print(f"Fresh Memory references: {ref}")
    # ref = sys.getrefcount(fresh_memory.memory)
    # print(f"Fresh Memory.memory references: {ref}")


    # return history, best, best_all


def save_csv_locked(df, path):
    while True:
        try:
            df.to_csv(path)
            break
        except PermissionError as er:
            RUN_LOGGER.warn(f"Waiting for file access ({path}): {er}")
            time.sleep(2)


def deep_q_reinforce_fresh(
        mod, fresh_samples,
        discount=0.9,
        env_alldata3d=None,
        mini_batchsize=500,
):
    RUN_LOGGER.debug(
            f"Trying to reinforce (fresh). MiniBatch:{mini_batchsize}. Dc: {discount}, samples: {len(fresh_samples)}")

    split_inds = get_splits(len(fresh_samples), 20000)
    shuffle(fresh_samples)

    losses = []
    time_f_start = time.time()

    for start_ind, stop_ind in zip(split_inds, split_inds[1:]):
        batch_time = time.time()
        RUN_LOGGER.debug(f"Training Batch: {start_ind, stop_ind}")
        samples_slice = fresh_samples[start_ind:stop_ind]

        envs_inds = []
        envs_inds_fut = []
        states = []
        states_fut = []
        actions = []
        rewards = []
        dones = []
        fresh_qvals = []

        "Make lists"
        for env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, rew, done, qvals in samples_slice:
            envs_inds.append(env_state_ind)
            if done:
                envs_inds_fut.append(env_state_ind)
                states_fut.append(agent_state)
                # envs_inds_fut.append(None)
            else:
                envs_inds_fut.append(env_state_ind_fut)
                states_fut.append(agent_state_fut)

            states.append(agent_state)
            actions.append(act)
            rewards.append(rew)
            dones.append(done)
            fresh_qvals.append(qvals)

        states = np.array(states)
        states_fut = np.array(states_fut)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        fresh_qvals = np.array(fresh_qvals)

        "Old States"
        # _wal, _st = np.array(old_states, dtype=object).T
        # _wal, _st = np.vstack(_wal), np.stack(_st)

        # q_vals_to_train = mod.predict([_wal, _st])

        "Future Q"
        # _ft_wal, _ft_sing = np.array(future_states, dtype=object).T
        # _ft_wal, _ft_sing = np.vstack(_ft_wal), np.stack(_ft_sing)
        envs_states_arr_fut = env_alldata3d[envs_inds_fut]
        envs_states_arr = env_alldata3d[envs_inds]

        max_future_argq = mod.predict([envs_states_arr_fut, states_fut], verbose=False).max(axis=1)

        new_qvals = sub_deepq_func(actions, discount, dones, fresh_qvals, max_future_argq, rewards)

        "Reinforce"
        time_pretrain = time.time()
        history_ob = mod.fit([envs_states_arr, states], new_qvals, shuffle=True,
                             batch_size=mini_batchsize,
                             verbose=False)

        timeend = time.time()
        RUN_LOGGER.debug(
                f"Trained (fresh) in: {timeend - batch_time:>5.4f}s. Fit duration:{timeend - time_pretrain:>5.4f}s, Loss: {history_ob.history['loss']}")

        losses.append(history_ob.history['loss'])
    timeend = time.time()
    RUN_LOGGER.info(f"Full (fresh) training took : {timeend - time_f_start :>6.3f}s")
    return np.mean(losses)


def sub_deepq_func(actions, discount, dones, curr_qvals, max_future_argq, rewards):
    # for qv_row, max_q, act, rew, done in zip(
    #         curr_qvals, max_future_argq, actions, rewards, dones):
    #     if done:
    #         qv_row[act] = rew
    #     else:
    #         targ = rew + discount * max_q * int(not done)
    #         qv_row[act] = targ

    new_vector = rewards + discount * max_future_argq * (1 - dones.astype(int))
    for i, (ac, v) in enumerate(zip(actions, new_vector)):
        curr_qvals[i, ac] = v

    return curr_qvals


def deep_q_reinforce_oldmem(
        mod, old_samples,
        discount=0.9,
        segments_of3ddata=None,
        mini_batchsize=500,
):
    # batch_gen = get_big_batch(
    #         fresh_mem, old_memory, big_batch, old_mem_fraction,
    #         min_batches=mini_batch)
    # fresh_mem: AgentsMemory
    # print("Shuffling:")
    # print(old_samples[:5])
    # shuffle(old_samples)
    # print(old_samples[:5])
    # print(f"Shuffled samples: {old_samples}")
    RUN_LOGGER.debug(
            f"Trying to reinforce (old). MiniBatch:{mini_batchsize}. Dc: {discount}, samples: {len(old_samples)}")
    # old_samples = old_memory.random_samples(0.99)
    # RUN_LOGGER.debug(f"Samples amount: {len(old_samples)}")

    split_inds = get_splits(len(old_samples), 20000)

    losses = []
    time_f_start = time.time()

    "BATCH TRAINING"
    for start_ind, stop_ind in zip(split_inds, split_inds[1:]):
        batch_time = time.time()
        RUN_LOGGER.debug(f"OldMemory Training Batch: {start_ind, stop_ind}")
        samples_slice = old_samples[start_ind:stop_ind]

        envs_inds = []
        envs_inds_fut = []
        states = []
        states_fut = []
        actions = []
        rewards = []
        dones = []

        "Make lists"
        for segmet_i, env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, rew, done in samples_slice:
            envs_inds.append((segmet_i, env_state_ind))
            if done:
                print("YOU GOT DONE SAMPLE")
                envs_inds_fut.append((segmet_i, env_state_ind))
                states_fut.append(agent_state)
                # envs_inds_fut.append(None)
            else:
                envs_inds_fut.append((segmet_i, env_state_ind_fut))
                states_fut.append(agent_state_fut)

            states.append(agent_state)
            actions.append(act)
            rewards.append(rew)
            dones.append(done)

        states = np.array(states)
        states_fut = np.array(states_fut)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        # fresh_qvals = np.array(fresh_qvals)


        "Get arrays from indexes"
        # envs_states_arr_fut = env_alldata3d[envs_inds_fut]
        # envs_states_arr = env_alldata3d[envs_inds]
        envs_states_arr_fut = [segments_of3ddata[segi][ind] for (segi, ind) in envs_inds_fut]
        envs_states_arr = [segments_of3ddata[segi][ind] for (segi, ind) in envs_inds]

        envs_states_arr_fut = np.array(envs_states_arr_fut)
        envs_states_arr = np.array(envs_states_arr)
        # print("Envs shapes now: cur/fut", envs_states_arr.shape, envs_states_arr_fut.shape)

        "Old States"
        # _wal, _st = np.array(old_states, dtype=object).T
        # _wal, _st = np.vstack(_wal), np.stack(_st)
        current_qvals = mod.predict([envs_states_arr, states], verbose=False)

        # q_vals_to_train = mod.predict([_wal, _st])

        "Future Q"
        # _ft_wal, _ft_sing = np.array(future_states, dtype=object).T
        # _ft_wal, _ft_sing = np.vstack(_ft_wal), np.stack(_ft_sing)

        max_future_argq = mod.predict([envs_states_arr_fut, states_fut], verbose=False).max(axis=1)
        new_qvals = sub_deepq_func(actions, discount, dones, current_qvals, max_future_argq, rewards)

        "Reinforce"
        time_pretrain = time.time()
        history_ob = mod.fit([envs_states_arr, states], new_qvals, shuffle=True,
                             batch_size=mini_batchsize,
                             verbose=False)

        timeend = time.time()
        RUN_LOGGER.debug(
                f"Training (old) in: {timeend - batch_time:>5.4f}s. Fit duration:{timeend - time_pretrain:>5.4f}s, Loss: {history_ob.history['loss']}")

        losses.append(history_ob.history['loss'])

    timeend = time.time()
    RUN_LOGGER.info(f"Full (old) training took : {timeend - time_f_start :>6.3f}s")
    return np.mean(losses)


# @measure_real_time_decorator
# def append_data_to_file(values, path_this_model_folder, file_name, axis=0):
#     if os.path.isfile(path_this_model_folder + file_name):
#         old_hist = np.load(path_this_model_folder + file_name, allow_pickle=True)
#         full_hist = np.concatenate([old_hist, values], axis=axis)
#         RUN_LOGGER.debug(f"Saved data to: {file_name}")
#     else:
#         full_hist = values
#         RUN_LOGGER.debug(f"Started saving data to: {file_name}")
#     np.save(path_this_model_folder + file_name, full_hist)


def get_big_batch(fresh_mem, old_mem, batch_s, old_mem_fr, min_batches):
    batches_n = np.ceil(len(fresh_mem) / batch_s).astype(int)

    old_mem_size = len(old_mem)

    for i in range(batches_n):
        frame1 = fresh_mem[i * batch_s:i * batch_s + batch_s]

        if len(frame1) >= min_batches:
            get_n_samples = int(len(frame1) * old_mem_fr)
            get_n_samples = get_n_samples if get_n_samples <= old_mem_size else old_mem_size
            frame2 = sample(old_mem, get_n_samples)

            yield [*zip(*frame1, *frame2)]


def single_model_training_function(
        counter, model_params, train_sequences, price_ind,
        games_n, game_duration,
        main_logger: logger
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
                loss, nodes, lr, iteration=iteration,
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
        train_qmodel(
                model, train_sequences,
                price_col_ind=price_ind,
                naming_ob=naming_ob,
                game_duration=game_duration,
                games_n=games_n,
                reward_f_num=reward_fnum,
                discount=discount,
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
            workers=10,
            minsamples_insegment=300,
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
    # gen1 = grid_models_generator(time_ftrs, time_size, float_feats=float_feats, out_size=out_sze)
    # gen2 = grid_models_generator_2(time_ftrs, time_size, float_feats=float_feats, out_size=output_size)
    gen_it23 = grid_models_generator_it23(time_ftrs, time_size, float_feats=float_feats,
                                          out_size=output_size)
    # gen1 = dummy_grid_generator()
    # for data in gen1:
    #     single_model_training_function(*data)
    # manager = mpc.Manager()
    # # manager.list()
    # manager.list(trainsegments_ofsequences3d)
    # print(manager)
    # manager.list([mpc.Array(segm) for segm in trainsegments_ofsequences3d])
    trainsegments_ofsequences3d = trainsegments_ofsequences3d[:40]

    with ProcessPoolExecutor(max_workers=4) as executor:
        process_list = []
        for counter, data in enumerate(gen_it23):
            MainLogger.info(f"Adding process with: {data}")
            games_n = 100
            game_duration = 500
            # if counter >= 2:
            #     break
            # elif counter <= 11:
            #     train_duration = 280
            # else:
            #     continue

            proc = executor.submit(
                    single_model_training_function, *data, trainsegments_ofsequences3d, price_ind,
                    games_n, game_duration,
                    MainLogger
            )
            process_list.append(proc)
            print(f"Added process: {counter}")

            # while True
        # proc.e
        #     results.append(proc)
        # print("Waiting:")
        # result = concurrent.futures.wait(process_list)
        # print("Waiting finished.")

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

    print("Script end....")
