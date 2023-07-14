import numpy as np

import time
import keras
import traceback
import sys
import os

from random import sample
from common_functions import NamingClass, get_splits
from actors import initialize_agents, resolve_actions

from common_settings import ITERATION, path_data_clean_folder, path_models
import logger
from reward_functions import RewardStore
from yasiu_native.time import measure_real_time_decorator
import pandas as pd


RUN_LOGGER = logger.logger
DEBUG_LOGGER = logger.debug_logger


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


class ModelMemory(AgentsMemory):
    def __init__(self, agents_mem: AgentsMemory = None):
        self.memory = []
        if agents_mem:
            self.migrate(agents_mem.memory)

    def add_sample(
            self, env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, reward, done,

    ):
        self.memory.append((
                env_state_ind, env_state_ind_fut, agent_state, agent_state_fut, act, reward, done,
        ))

    def migrate(self, mem):
        for batch in mem:
            # print(f"Migrating batch: {batch}")
            self.add_sample(*batch[:-1])  # Drop q-vals at end

    def __str__(self):
        return f"ModelMemory (Size 6 lists each: {len(self.memory)})"


def get_eps(n, epoch_max, repeat=5, eps_power=3, max_explore=0.8):
    # f2 = 1 / repeat
    val = (1 - (np.mod(n / epoch_max, 1 / repeat) * repeat) ** eps_power) * max_explore
    return val


def train_qmodel(
        model_keras: keras.Model, naming_ob: NamingClass,
        datalist_2dsequences_ordered_train, price_col_ind,
        fulltrain_ntimes=1000,
        agents_n=10,
        session_size=3600,

        allow_train=True, fresh_memory: ModelMemory = None,

        # Optional
        max_eps=0.8, override_eps=None,
        old_mem_fraction=0.2,
        fresh_mem_fraction=0.7,
        # refresh_n_times=3,
        # local_minima=None, local_maxima=None,
        # local_minima_soft=None, local_maxima_soft=None,
        reward_f_num=1,
        discount=0.98,
        # director_loc=None, name=None, timeout=None,
        # save_qval_dist=False,

        # PARAMS ==============
        time_window_size=10,
        stock_price_multiplier=0.5,
        stock_ammount_in_bool=True,
        # reward_fun: reward_fun_template,
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

    os.makedirs(path_this_model_folder, exist_ok=True)
    os.makedirs(os.path.join(path_this_model_folder + "data"), exist_ok=True)

    if os.path.isfile(path_this_model_folder + "weights.keras"):
        RUN_LOGGER.info(f"Loading model: {path_this_model_folder}")
        model_keras.load_weights(path_this_model_folder + "weights.keras")
    else:
        RUN_LOGGER.info("Not loading model.")
    if fresh_memory is None:
        # memory = deque(maxlen=300_000)
        # fresh_memory = AgentsMemory()
        agents_memory = AgentsMemory()
    else:
        assert isinstance(fresh_memory, (AgentsMemory,)), "Memory must be an instance of ModelMemory"

    if not allow_train:
        agents_n = 1

    if N_SAMPLES <= 0:
        raise ValueError(
                f"Too few samples! {N_SAMPLES}, shape:{datalist_2dsequences_ordered_train.shape}")

    reward_fun = RewardStore.get(reward_f_num)

    out_size = int(naming_ob.outsize)
    qv_dataframe = pd.DataFrame(columns=['eps', 'sess_i', 'sample_n', 'buy', 'idle', 'sell'])
    session_dataframe = pd.DataFrame(columns=['eps', 'session_num', 'ind_start', 'ind_end', 'gain'])

    LOOP_TIMES = []

    for i_train_sess in range(fulltrain_ntimes):
        last_start = N_SAMPLES - session_size
        # print(f"Last start: {last_start} for: {N_SAMPLES} of size: {session_size}")
        ses_start = np.random.randint(0, last_start)
        ses_end = ses_start + session_size

        RUN_LOGGER.debug(
                f"Staring session: {i_train_sess + 1} of {fulltrain_ntimes} : {naming_ob.path} Indexes: {ses_start, ses_end}, Ses:size: {session_size}")
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

        QHISTORY = []

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
        for i_sample in range(ses_start, ses_end):
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
            # elif FORCE_RANDOM:
            #     actions = np.random.randint(0, 3, agents_n)
            else:
                actions = np.argmax(q_vals, axis=-1)
                # QHISTORY.append(q_vals.ravel())
                for qv in q_vals:
                    qv_dataframe.loc[len(qv_dataframe)] = session_eps, i_train_sess, i_sample, *qv

            # last_price = prices[x]
            rewards = []
            valids = []

            env_state_arr = timesegment_2d

            cur_step_price = env_state_arr[0, price_col_ind]
            for state_arr, act, hidden_arr in zip(agents_discrete_states, actions, hidden_states):
                # hidden_arr[1] = cur_step_price

                rew, valid = reward_fun(
                        env_state_arr, state_arr, act, hidden_arr, done_session, cur_step_price,
                )
                rewards.append(rew)
                valids.append(valid)

            if allow_train:
                env_states_inds = [i_sample] * agents_n
                env_states_inds_fut = [i_sample + 1] * agents_n
                # agents_states = agents_discrete_states.copy()
                new_states, new_hidden_states = resolve_actions(
                        cur_step_price, agents_discrete_states, hidden_states, actions
                )

                dones = [done_session] * agents_n

                fresh_memory.add_batch(env_states_inds, env_states_inds_fut, agents_discrete_states,
                                       new_states,
                                       actions, rewards, dones, q_vals)
                # print(fresh_memory)
            else:
                "Dont train"
                new_states, new_hidden_states = resolve_actions(
                        cur_step_price, agents_discrete_states, hidden_states, actions
                )

            agents_discrete_states = new_states
            hidden_states = new_hidden_states

        tend_walking = time.time()
        RUN_LOGGER.info(
                f"Walking through data took: {tend_walking - t0_walking:>5.4f}s. {(tend_walking - t0_walking) / N_SAMPLES:>5.5f}s per step")

        gain = hidden_states[:, 0] - hidden_states[:, 1]
        # RUN_LOGGER.debug(f"End gains: {gain}")
        for g in gain:
            session_dataframe.loc[
                len(session_dataframe)] = session_eps, i_train_sess, ses_start, ses_end, g
        DEBUG_LOGGER.debug(hidden_states)
        DEBUG_LOGGER.debug(f"End gains rel: {(gain / hidden_states[:, 1]).reshape(-1, 1)}")
        # RUN_LOGGER.debug(f"End cargo: {hidden_states[:, 2]}")

        "Session Training"
        if allow_train:
            loss = deep_q_reinforce(
                    model_keras, fresh_memory,
                    discount=discount,
                    env_data_2d=datalist_2dsequences_ordered_train,
                    mini_batchsize=int(naming_ob.batch),
            )
            # L(history.history['loss'])
            append_data_to_file([loss], path_this_model_folder, "hist_loss.npy")

        "RESOLVE END SCORE"

        endtime = time.time()
        duration = endtime - starttime
        LOOP_TIMES.append(duration)
        DEBUG_LOGGER.info(
                f"This loop took: {duration:>5.4f}s. Current memory samples: {len(fresh_memory.memory)}")
        RUN_LOGGER.info(
                f"This loop took: {duration:>5.4f}s. Current memory samples: {len(fresh_memory.memory)}")

        if allow_train and not i_train_sess % 10:
            model_keras.save_weights(path_this_model_folder + "weights.keras")
            RUN_LOGGER.info(f"Saved weights: {naming_ob}")
            save_csv_locked(session_dataframe,
                            os.path.join(path_this_model_folder, "data", "session.csv"))
            save_csv_locked(qv_dataframe, os.path.join(path_this_model_folder, "data", "qvals.csv"))

    if allow_train:
        model_keras.save_weights(path_this_model_folder + "weights.keras")
        RUN_LOGGER.info(f"Saved weights: {naming_ob}")
        save_csv_locked(session_dataframe,
                        os.path.join(path_this_model_folder, "data", "session.csv"))
        save_csv_locked(qv_dataframe, os.path.join(path_this_model_folder, "data", "qvals.csv"))

    append_data_to_file(LOOP_TIMES, path_this_model_folder, "hist_times.npy")
    DEBUG_LOGGER.debug(f"Mean loop time = {np.mean(LOOP_TIMES) / 60:4.4f} m")
    RUN_LOGGER.info(f"Mean loop time = {np.mean(LOOP_TIMES) / 60:4.4f} m")

    # return history, best, best_all


def save_csv_locked(df, path):
    while True:
        try:
            df.to_csv(path)
            break
        except PermissionError as er:
            RUN_LOGGER.warn(f"Waiting for file access ({path}): {er}")
            time.sleep(2)


def deep_q_reinforce(
        mod, fresh_mem,
        discount=0.9,
        env_data_2d=None,
        mini_batchsize=500,
):
    # batch_gen = get_big_batch(
    #         fresh_mem, old_memory, big_batch, old_mem_fraction,
    #         min_batches=mini_batch)
    # fresh_mem: AgentsMemory
    RUN_LOGGER.debug(f"Trying to reinforce. MiniBatch:{mini_batchsize}. Dc: {discount}")
    fresh_samples = fresh_mem.random_samples(0.99)
    RUN_LOGGER.debug(f"Samples amount: {len(fresh_samples)}")

    split_inds = get_splits(len(fresh_samples), 50000)

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

        for fresh_qrow, max_q, act, rew, done in zip(fresh_qvals, max_future_argq, actions, rewards,
                                                     dones):
            if done:
                # targ = rew
                fresh_qrow[2] = rew
            else:
                targ = rew + discount * max_q * int(not done)
                fresh_qrow[act] = targ
            # print(f"{row}, {act}, rew:{rew:8.3f},  maxq:{max_q:6.2f}, r+g*max:{targ:5.2f}, {done}")

        "Reinforce"
        time_pretrain = time.time()
        history_ob = mod.fit([envs_states_arr, states], fresh_qvals, shuffle=True,
                             batch_size=mini_batchsize,
                             verbose=False)

        timeend = time.time()
        RUN_LOGGER.debug(
                f"Trained in: {timeend - batch_time:>5.4f}s. Fit duration:{timeend - time_pretrain:>5.4f}s, Loss: {history_ob.history['loss']}")

        losses.append(history_ob.history['loss'])
    timeend = time.time()
    RUN_LOGGER.info(f"Full training took : {timeend - time_f_start :>6.3f}s")
    return np.mean(losses)


# @measure_real_time_decorator
def append_data_to_file(values, path_this_model_folder, file_name, axis=0):
    if os.path.isfile(path_this_model_folder + file_name):
        old_hist = np.load(path_this_model_folder + file_name, allow_pickle=True)
        full_hist = np.concatenate([old_hist, values], axis=axis)
        RUN_LOGGER.debug(f"Saved data to: {file_name}")
    else:
        full_hist = values
        RUN_LOGGER.debug(f"Started saving data to: {file_name}")
    np.save(path_this_model_folder + file_name, full_hist)


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


from model_creator import grid_models_generator
from common_functions import to_sequences_forward


if __name__ == "__main__":
    # memory_ob = ModelMemory()
    #
    # x = np.linspace(0, 1, 10001)
    # y = get_eps(x, 1)
    # plt.plot(x, y)
    # plt.show()
    # train_qmodel()
    # agents_states, initial_states, empty_ref_list = initialize_agents(10)
    # print(agents_states)
    # print(empty_ref_list)
    # empty_ref_list[0] = [1]
    # empty_ref_list[1].append(2)
    # print(empty_ref_list)
    # print(initial_states)
    RUN_LOGGER.info("=== NEW TRAINING ===")

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
    # _, model, (arch_num, loss, nodes, batch, lr) = next(gen1)

    for counter, model, (arch_num, loss, nodes, batch, lr) in gen1:
        RUN_LOGGER.info(
                f"Starting {counter}: Arch Num:{arch_num} Version:? Loss:{loss} Nodes:{nodes} Batch:{batch} Lr:{lr}")
        naming_ob = NamingClass(
                arch_num, ITERATION,
                time_feats=time_ftrs, time_window=time_wind, float_feats=float_feats, outsize=out_sze,
                node_size=nodes,

                learning_rate=lr, loss=loss, batch=batch
        )

        try:
            train_qmodel(model, naming_ob, train_sequences, price_col_ind=price_id, session_size=3600)
        except Exception as exc:
            import traceback


            "PRINT TO SYS"
            RUN_LOGGER.error(exc, exc_info=True)
            # print(exc.__traceback__.msg)

            break
    # memory = AgentsMemory()
    # memory.add_sample(1, 2, 3, 4, 5, 6, 7, 8)
    #
    # memory.add_batch(*np.random.random((8, 5)))
    #
    # print(memory)
    # # print(memory.)
    # samples = memory.random_samples()
    # # print("Random samples:")
    # # print(samples)
    #
    # memory2 = ModelMemory(memory)
    # print(memory2)
