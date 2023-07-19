import numpy as np

import time
from tensorflow import keras
import os
import datetime
import logger

from random import sample, shuffle
from actors import initialize_agents, resolve_actions_multibuy, resolve_actions_singlebuy

from common_settings import ITERATION, path_data_clean_folder, path_models
from common_functions import NamingClass, get_splits, get_eps, to_sequences_forward
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
            self, env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, reward, done,
    ):
        self.memory.append((
                env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, reward, done,
        ))

    @measure_real_time_decorator
    def migrate(self, mem):
        for batch in mem:
            # print(f"Migrating batch: {batch}")
            self.add_sample(*batch[:-1])  # Drop q-vals at end

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

        with open(path_qvals, "at", buffering=1) as q_textfile:
            arr = ['sess_eps', 'i_train_sess', 'agent_i', 'i_sample', 'q1', 'q2', 'q3']
            q_textfile.write(','.join(arr) + "\n")

            with open(path_sess, "at", buffering=1) as sess_textfile:
                arr2 = ['sess_eps', 'i_train_sess', 'sess_start', 'sess_end', 'agent_i', 'gain']
                sess_textfile.write(','.join(arr2) + "\n")

                with open(path_times, "at", buffering=1) as time_textfile:
                    time_textfile.write("loop_time\n")
                    with open(path_loss, "at", buffering=1) as loss_textfile:
                        loss_textfile.write("sess_i,session_meanloss\n")
                        with open(path_rewards, "at", buffering=1) as rew_textfile:
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
        model_keras: keras.Model, datalist_2dsequences_ordered_train,
        price_col_ind,
        naming_ob: NamingClass,
        fulltrain_ntimes=1000,
        agents_n=10,
        session_size=3600,

        allow_train=True, model_memory: ModelMemory = None,

        # Optional
        max_eps=0.8, override_eps=None,
        remember_fresh_fraction=0.3,
        train_from_oldmem_fraction=0.4,
        old_memory_size=400_000,
        # refresh_n_times=3,
        # local_minima=None, local_maxima=None,
        # local_minima_soft=None, local_maxima_soft=None,
        reward_f_num=2,
        discount=0.95,
        # director_loc=None, name=None, timeout=None,
        # save_qval_dist=False,

        # PARAMS ==============
        # time_window_size=10,
        # stock_price_multiplier=0.5,
        # stock_ammount_in_bool=True,
        # reward_fun: reward_fun_template,
        allow_multibuy=False,

        # FILE SAVERS
        qvals_file: TextIOWrapper = None,
        session_file: TextIOWrapper = None,
        time_file: TextIOWrapper = None,
        loss_file: TextIOWrapper = None,
        rew_file: TextIOWrapper = None,
):
    RUN_LOGGER.debug(
            f"Train params: {naming_ob}: trainN:{fulltrain_ntimes}, agents: {agents_n}. Reward F:{reward_f_num}")
    # N_SAMPLES = (data_list_2darr.shape[0] - time_frame)
    N_SAMPLES = len(datalist_2dsequences_ordered_train)
    WALK_INTERVAL_DEBUG = 250
    RUN_LOGGER.debug(f"Input samples: {datalist_2dsequences_ordered_train.shape}")
    # print(normed_data)
    # print(normed_data.shape)
    path_this_model_folder = os.path.join(path_models, naming_ob.path, "")

    if os.path.isfile(path_this_model_folder + "weights.keras"):
        RUN_LOGGER.info(f"Loading model: {path_this_model_folder}")
        model_keras.load_weights(path_this_model_folder + "weights.keras")
    else:
        RUN_LOGGER.info("Not loading model.")

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

    if N_SAMPLES <= 0:
        raise ValueError(
                f"Too few samples! {N_SAMPLES}, shape:{datalist_2dsequences_ordered_train.shape}")

    reward_fun = RewardStore.get(reward_f_num)

    out_size = int(naming_ob.outsize)
    # qv_dataframe = pd.DataFrame(columns=['eps', 'sess_i', 'sample_n', 'buy', 'idle', 'sell'])
    # session_dataframe = pd.DataFrame(columns=['eps', 'session_num', 'ind_start', 'ind_end', 'gain'])

    LOOP_TIMES = deque(maxlen=100)

    for i_train_sess in range(fulltrain_ntimes):
        last_start = N_SAMPLES - session_size
        # print(f"Last start: {last_start} for: {N_SAMPLES} of size: {session_size}")
        ses_start = np.random.randint(0, last_start)
        ses_end = ses_start + session_size

        starttime = time.time()
        "Clone model if step training"
        # stable_model = model_keras.copy()

        # eps_cycle = EPS_CYCLE + np.random.randint(0, 100)
        # eps_offset = np.random.randint(0, 50)
        fresh_memory = AgentsMemory()

        if override_eps is not None:
            session_eps = override_eps
        else:
            session_eps = get_eps(i_train_sess, fulltrain_ntimes, max_explore=max_eps)

        RUN_LOGGER.debug(
                f"Staring session: {i_train_sess + 1} of {fulltrain_ntimes} (Eps: {session_eps:>2.3f}) : {naming_ob.path} Indexes: {ses_start, ses_end}, Ses:size: {session_size}.")

        "Start with different money values"
        # "CASH | CARGO | LAST SELL | LAST BUY | LAST TRANSACTION PRICE"

        if allow_train:
            # agents_stocks, current_cash_arr, starting_money_arr = initialize_agents(START, agents_n)
            if True:
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
        for i_sample in range(ses_start, ses_end - 1):  # Never in done state
            done_session = i_sample == (N_SAMPLES - 1)  # is this last sample?

            timesegment_2d = datalist_2dsequences_ordered_train[i_sample, :]
            timesegment_stacked = np.tile(timesegment_2d[np.newaxis, :, :], (agents_n, 1, 1))
            if not i_sample % WALK_INTERVAL_DEBUG:
                RUN_LOGGER.debug(f"Walking sample: {i_sample}, memory: {len(fresh_memory.memory)}")

            if done_session:
                future_segment3d = None
            else:
                future_segment3d = datalist_2dsequences_ordered_train[i_sample + 1, :][np.newaxis, :, :]
                futuresegment_stacked = np.tile(future_segment3d, (agents_n, 1, 1))

            "MOVE TO IF BELOW"
            # print("Discrete shape", agents_discrete_states.shape)
            # print(f"Predicted: {q_vals}")
            # print(q_vals.shape)

            q_vals = model_keras.predict(
                    [timesegment_stacked, agents_discrete_states],
                    verbose=False
            )
            "Select Action"
            if allow_train and session_eps > 0 and session_eps > np.random.random():
                "Random Action"
                actions = np.random.randint(0, out_size, agents_n)
                # print(f"Random actions: {actions}")
            # elif FORCE_RANDOM:
            #     actions = np.random.randint(0, 3, agents_n)
            else:
                actions = np.argmax(q_vals, axis=-1)

                "Saving predicted qvals"
                for agent_i, qv in enumerate(q_vals):
                    # qv_dataframe.loc[len(qv_dataframe)] = session_eps, i_train_sess, i_sample, *qv
                    arr = [session_eps, i_train_sess, agent_i, i_sample, *qv]
                    text = ','.join([str(a) for a in arr]) + "\n"
                    qvals_file.write(text)

            # last_price = prices[x]
            rewards = []
            valids = []

            env_state_arr = timesegment_2d

            cur_step_price = env_state_arr[0, price_col_ind]
            for agent_i, (state_arr, act, hidden_arr) in enumerate(
                    zip(agents_discrete_states, actions, hidden_states)
            ):
                # hidden_arr[1] = cur_step_price

                rew, valid = reward_fun(
                        env_state_arr, state_arr, act, hidden_arr, done_session, cur_step_price,
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

                fresh_memory.add_batch(env_states_inds, env_states_inds_fut, agents_discrete_states,
                                       new_states,
                                       actions, rewards, dones, q_vals)

            else:
                "Dont train"
                new_states, new_hidden_states = resolve_actions_func(
                        cur_step_price, agents_discrete_states, hidden_states, actions
                )

            agents_discrete_states = new_states
            hidden_states = new_hidden_states

        tend_walking = time.time()
        RUN_LOGGER.info(
                f"Walking through data took: {tend_walking - t0_walking:>5.4f}s. {(tend_walking - t0_walking) / N_SAMPLES:>5.5f}s per step")

        gain = hidden_states[:, 0] - hidden_states[:, 1]
        # RUN_LOGGER.debug(f"End gains: {gain}")

        "Saving session data"
        for gi, g in enumerate(gain):
            arr = [session_eps, i_train_sess, ses_start, ses_end, gi, g]
            text = ','.join([str(a) for a in arr]) + "\n"
            # print(f"Writing text to session: '{text}'")
            session_file.write(text)

        DEBUG_LOGGER.debug(hidden_states)
        DEBUG_LOGGER.debug(f"End gains: {gain.reshape(-1, 1)}")
        DEBUG_LOGGER.debug(f"End cargo: {hidden_states[:, 2].reshape(-1, 1)}")
        # RUN_LOGGER.debug(f"End cargo: {hidden_states[:, 2]}")

        "Session Training"
        if allow_train:
            loss = deep_q_reinforce_fresh(
                    model_keras, fresh_memory.memory,
                    discount=discount,
                    env_data_2d=datalist_2dsequences_ordered_train,
                    mini_batchsize=int(naming_ob.batch),
            )
            # L(history.history['loss'])
            loss_file.write(f"{i_train_sess},{loss}\n")

            k = int(remember_fresh_fraction * len(fresh_memory))
            model_memory.migrate(sample(fresh_memory.memory, k))

            k = int(train_from_oldmem_fraction * len(model_memory))
            if k > 200:
                old_samples = sample(model_memory.memory, k)
                loss = deep_q_reinforce_oldmem(
                        model_keras, old_samples,
                        discount=discount,
                        env_data_2d=datalist_2dsequences_ordered_train,
                        mini_batchsize=int(naming_ob.batch),
                )
                loss_file.write(f"{i_train_sess},{loss}\n")

        "RESOLVE END SCORE"

        endtime = time.time()
        duration = endtime - starttime
        LOOP_TIMES.append(duration)
        loop_sum_text = f"This loop took: {duration:>5.4f}s. Fresh mem: {len(fresh_memory)}, Total mem: {len(model_memory)}"
        DEBUG_LOGGER.info(loop_sum_text)
        RUN_LOGGER.info(loop_sum_text)

        if allow_train and not i_train_sess % 10:
            model_keras.save_weights(path_this_model_folder + "weights.keras")
            RUN_LOGGER.info(f"Saved weights: {naming_ob}")

        time_file.write(f"{duration}\n")
        DEBUG_LOGGER.debug(f"Mean loop time = {np.mean(LOOP_TIMES) / 60:4.4f} m")
        RUN_LOGGER.info(f"Mean loop time = {np.mean(LOOP_TIMES) / 60:4.4f} m")

    if allow_train:
        model_keras.save_weights(path_this_model_folder + "weights.keras")
        RUN_LOGGER.info(f"Saved weights: {naming_ob}")


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
        env_data_2d=None,
        mini_batchsize=500,
):
    RUN_LOGGER.debug(
            f"Trying to reinforce (fresh). MiniBatch:{mini_batchsize}. Dc: {discount}, samples: {len(fresh_samples)}")

    split_inds = get_splits(len(fresh_samples), 50000)
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
        rewards = np.array(actions)
        dones = np.array(dones)
        fresh_qvals = np.array(fresh_qvals)

        "Old States"
        # _wal, _st = np.array(old_states, dtype=object).T
        # _wal, _st = np.vstack(_wal), np.stack(_st)

        # q_vals_to_train = mod.predict([_wal, _st])

        "Future Q"
        # _ft_wal, _ft_sing = np.array(future_states, dtype=object).T
        # _ft_wal, _ft_sing = np.vstack(_ft_wal), np.stack(_ft_sing)
        envs_states_arr_fut = env_data_2d[envs_inds_fut]
        envs_states_arr = env_data_2d[envs_inds]

        max_future_argq = mod.predict([envs_states_arr_fut, states_fut]).max(axis=1)

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
    # print("New vector")
    # print(new_vector[:10])
    # print(new_vector.shape)
    # curr_qvals[:, actions] = new_vector
    for i, (ac, v) in enumerate(zip(actions, new_vector)):
        curr_qvals[i, ac] = v

    return curr_qvals


def deep_q_reinforce_oldmem(
        mod, old_samples,
        discount=0.9,
        env_data_2d=None,
        mini_batchsize=500,
):
    # batch_gen = get_big_batch(
    #         fresh_mem, old_memory, big_batch, old_mem_fraction,
    #         min_batches=mini_batch)
    # fresh_mem: AgentsMemory
    # print("Shuffling:")
    # print(old_samples[:5])
    shuffle(old_samples)
    # print(old_samples[:5])
    # print(f"Shuffled samples: {old_samples}")
    RUN_LOGGER.debug(
            f"Trying to reinforce (old). MiniBatch:{mini_batchsize}. Dc: {discount}, samples: {len(old_samples)}")
    # old_samples = old_memory.random_samples(0.99)
    # RUN_LOGGER.debug(f"Samples amount: {len(old_samples)}")

    split_inds = get_splits(len(old_samples), 50000)

    losses = []
    time_f_start = time.time()

    for start_ind, stop_ind in zip(split_inds, split_inds[1:]):
        batch_time = time.time()
        RUN_LOGGER.debug(f"Training Batch: {start_ind, stop_ind}")
        samples_slice = old_samples[start_ind:stop_ind]

        envs_inds = []
        envs_inds_fut = []
        states = []
        states_fut = []
        actions = []
        rewards = []
        dones = []

        "Make lists"
        for env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, rew, done in samples_slice:
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

        states = np.array(states)
        states_fut = np.array(states_fut)
        actions = np.array(actions)
        rewards = np.array(actions)
        dones = np.array(dones)
        # fresh_qvals = np.array(fresh_qvals)


        "Get arrays from indexes"
        envs_states_arr_fut = env_data_2d[envs_inds_fut]
        envs_states_arr = env_data_2d[envs_inds]

        "Old States"
        # _wal, _st = np.array(old_states, dtype=object).T
        # _wal, _st = np.vstack(_wal), np.stack(_st)
        current_qvals = mod.predict([envs_states_arr, states])

        # q_vals_to_train = mod.predict([_wal, _st])

        "Future Q"
        # _ft_wal, _ft_sing = np.array(future_states, dtype=object).T
        # _ft_wal, _ft_sing = np.vstack(_ft_wal), np.stack(_ft_sing)


        max_future_argq = mod.predict([envs_states_arr_fut, states_fut]).max(axis=1)
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


def load_data_split(path, train_split=0.65, ):
    arr = np.load(path, allow_pickle=True)
    # df = pd.read_csv(path)

    pivot = int(len(arr) * train_split)

    df_train = arr[:pivot, :]
    df_test = arr[pivot:, :]
    print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test


def single_model_training_function(
        counter, model_params, train_sequences, price_id,
        main_logger: logger
):
    "LIMIT GPU BEFORE BUILDING MODEL"
    main_logger.info(f"Process of: {counter} has started now.")

    "GLOBAL LOGGERS"
    global RUN_LOGGER
    global DEBUG_LOGGER
    RUN_LOGGER = logger.create_logger(number=counter)
    DEBUG_LOGGER = logger.create_debug_logger(number=counter)

    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        (arch_num, time_feats, time_window, float_feats, out_size,
         nodes, lr, batch, loss
         ) = model_params
        model = model_builder(
                arch_num, time_feats, time_window, float_feats, out_size, loss, nodes, lr,
                batch
        )
        reward_fnum = 2

        RUN_LOGGER.info(
                f"Starting {counter}: Arch Num:{arch_num} Version:? Loss:{loss} Nodes:{nodes} Batch:{batch} Lr:{lr}")
        naming_ob = NamingClass(
                arch_num, ITERATION,
                time_feats=time_feats, time_window=time_window, float_feats=float_feats,
                outsize=out_size,
                node_size=nodes, reward_fnum=reward_fnum,

                learning_rate=lr, loss=loss, batch=batch,
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
                price_col_ind=price_id,
                naming_ob=naming_ob,
                session_size=1500,
                fulltrain_ntimes=300,
                reward_f_num=reward_fnum,
        )
    except Exception as exc:
        "PRINT TO SYS"
        print(f"EXCEPTION: {exc}")
        RUN_LOGGER.error(exc, exc_info=True)

    "Clear memory?"
    del model


if __name__ == "__main__":
    MainLogger = logger.create_logger(name="MainProcess")
    # DEBUG_LOGGER = logger.debug_logger
    MainLogger.info("=== NEW TRAINING ===")

    "LOAD Data"
    columns = np.load(path_data_clean_folder + "int_norm.columns.npy", allow_pickle=True)
    print(
            "Loading file with columns: ", columns,
    )
    price_id = np.argwhere(columns == "last").ravel()[0]
    print(f"Price `last` at col: {price_id}")
    train_data, test_data = load_data_split(path_data_clean_folder + "int_norm.arr.npy")

    time_wind = 60
    float_feats = 1
    out_sze = 3
    train_sequences, _ = to_sequences_forward(train_data[:, :], time_wind, [1])

    samples_n, _, time_ftrs = train_sequences.shape
    print(f"Train sequences shape: {train_sequences.shape}")

    "Model Grid"
    gen1 = grid_models_generator(time_ftrs, time_wind, float_feats=float_feats, out_size=out_sze)
    # gen1 = dummy_grid_generator()
    # for data in gen1:
    #     single_model_training_function(*data)

    with ProcessPoolExecutor(max_workers=4) as executor:
        process_list = []
        for counter, data in enumerate(gen1):
            MainLogger.info(f"Adding process with: {data}")
            proc = executor.submit(single_model_training_function, *data, train_sequences, price_id,
                                   MainLogger)
            process_list.append(proc)
            counter += 1
            if counter > 4:
                break

            # while True
        # proc.e
        #     results.append(proc)
        print("Waiting:")
        # result = concurrent.futures.wait(process_list)
        # print("Waiting finished.")
        for proc in process_list:
            print(f"Waiting for Proc {proc}")
            proc.result()
            print(f"Proc {proc} has finished.")

        # for f in as_completed(process_list):
        #     print(f.result())

    # for res in process_list:
    #     print(res)
    #     res.join()
