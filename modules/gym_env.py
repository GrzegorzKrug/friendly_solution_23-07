from arg_parser import read_arguments_baseline
import time
import glob


exec_arguments = read_arguments_baseline()

import pandas as pd
import os
import sys


module_path = './gym_anytrading'
if not os.path.exists(module_path):
    os.chdir('../')

from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt

import gym
from gym import spaces
import numpy as np

from common_settings import path_data_folder, path_models, path_models_baseline
from preprocess_data import preprocess_pipe_bars

from matplotlib.style import use
from matplotlib.colors import CSS4_COLORS
import random


class TradingEnvironment(gym.Env):
    def __init__(self, segments_list, columns, reward_func=1):
        super(TradingEnvironment, self).__init__()

        self.segments_list = segments_list
        self.columns = columns
        # print(columns)
        self.price_col_ind = np.argwhere(columns == 'last').ravel()[0]
        self.timediff_col_ind = np.argwhere(columns == 'timediff_s').ravel()[0]
        # print(f"Timediff col: {self.timediff_col_ind}")

        # print(segments_list[0].shape)
        # print(segments_list[0][:50, -1, self.timediff_col_ind])

        # self.max_steps = len(self.df)

        OSB_SIZE = np.prod(segments_list[0].shape[1:]) + 1
        # print(segments_list[0].shape)
        # print(f"OBS SIZE: {OSB_SIZE}")

        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Pass, 2: Sell
        self.observation_space = spaces.Box(
                low=-10, high=10, shape=(OSB_SIZE,),
                dtype=np.float32
        )
        self.reward_action_cost = 0.001
        self.score_action_cost = 0.0001

        if reward_func == 1:
            self.step = self._step_rew_1
        elif reward_func == 2:
            self.step = self._step_rew_2
        else:
            raise ValueError(f"Not supported reward func: {reward_func}")

    def reset(self, segm_i=None):
        if segm_i is None:
            self.segm_i = np.random.randint(0, len(self.segments_list))
        else:
            self.segm_i = segm_i

        self.current_step = 0
        self.max_steps = len(self.segments_list[self.segm_i]) - 1
        self.hist_actions = []
        self.state = 0
        self.action_counter = 0
        self.idle_counter = 0
        self.buy_price = 0
        self.cash = self.segments_list[self.segm_i][0, -1, self.price_col_ind]
        self.score = self.cash
        self.start_cash = self.cash
        # self.value_hist=[]
        # self.reward_hist=[]

        print(f"RESETING: segm_i: {self.segm_i}")

        return self.observation

    @property
    def observation(self):
        vec = self.segments_list[self.segm_i][self.current_step].ravel()
        return np.concatenate([vec, [self.state]])

    def _step_rew_1(self, action):
        # print(f"Step: {self.current_step}-{action} (max: {self.max_steps})")

        assert self.action_space.contains(action)
        price = self.segments_list[self.segm_i][self.current_step, -1, self.price_col_ind]
        # fut_price = self.segments_list[self.segm_i][self.current_step + 1, -1, self.price_col_ind]
        # price = time_arr[-1, self.price_col_ind]
        self.idle_counter += self.segments_list[self.segm_i][
            self.current_step, -1, self.timediff_col_ind]

        if action == 0:
            if self.state == 0:
                reward = -price - self.reward_action_cost
                self.buy_price = price
                self.state = 1

                self.action_counter += 1
                self.idle_counter = 0

                self.score -= self.score_action_cost
                self.cash = self.cash - price - self.score_action_cost
                was_valid = True
            else:
                reward = -10
                was_valid = False


        elif action == 1:
            was_valid = True
            rising_bar = (
                    self.segments_list[self.segm_i][self.current_step, -1, self.price_col_ind] \
                    - np.max(self.segments_list[self.segm_i][self.current_step, -5:-1,
                             self.price_col_ind])
            )  # Pos = Rising
            reward = np.clip(-rising_bar * 500, -1, 1)
            # diff = fut_price - price
            # reward=0
            # if self.state == 1:
            #     reward = diff * 100
            # else:
            #     reward = -diff * 100

        else:
            if self.state == 1:
                gain = price - self.buy_price
                # reward = price - action_cost
                timediff = self.segments_list[self.segm_i][self.current_step, -1, self.timediff_col_ind]
                bars_distance_scaling = np.clip(timediff, 0.2, 2)  # (0.2 , 1.5, 2.28, 0.8,) ** 2

                quick_sell_penalty = -3 / self.idle_counter / bars_distance_scaling

                rew = gain * 4000 - self.reward_action_cost
                # rew = np.clip(rew, -3, 3)
                # print(reward, quick_sell_penalty)
                reward = rew + quick_sell_penalty
                # print(reward, rew, quick_sell_penalty)

                self.state = 0
                self.idle_counter = 0
                self.action_counter += 1

                self.cash = self.cash + price - self.score_action_cost
                self.score = self.cash
                was_valid = True
            else:
                reward = -10
                was_valid = False

        # price = self.df['price'].iloc[self.current_step]
        # reward = price * 10

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done and self.action_counter < 10:
            reward = -20

        elif done and self.state == 1:
            reward = -3
        # elif done:
        #     reward = 0

        reward -= self.idle_counter / 300

        if was_valid:
            reward = np.clip(reward, -9, 9)
        else:
            reward = np.clip(reward, -10, 10)
        # if done:
        #     return np.array([price]), reward, done, {}

        # next_price = self.df['price'].iloc[self.current_step]
        next_observation = self.observation

        return next_observation, reward, done, {}

    def _step_rew_2(self, action):
        # print(f"Step: {self.current_step}-{action} (max: {self.max_steps})")

        assert self.action_space.contains(action)
        price = self.segments_list[self.segm_i][self.current_step, -1, self.price_col_ind]
        # fut_price = self.segments_list[self.segm_i][self.current_step + 1, -1, self.price_col_ind]
        # price = time_arr[-1, self.price_col_ind]
        self.idle_counter += self.segments_list[self.segm_i][
            self.current_step, -1, self.timediff_col_ind]

        rising_bar = (
                self.segments_list[self.segm_i][self.current_step, -1, self.price_col_ind] \
                - np.max(self.segments_list[self.segm_i][self.current_step, -5:-1,
                         self.price_col_ind])
        )  # Pos = Rising
        if action == 0:
            if self.state == 0:
                reward = -price - self.reward_action_cost - np.clip(rising_bar * 10, -0.5, 0.5)
                self.buy_price = price
                self.state = 1

                self.action_counter += 1
                self.idle_counter = 0

                self.score -= self.score_action_cost
                self.cash = self.cash - price - self.score_action_cost
                was_valid = True
            else:
                reward = -10
                was_valid = False

        elif action == 1:
            was_valid = True
            reward = np.clip(-rising_bar * 1000, -2, 2)

        else:
            if self.state == 1:
                gain = price - self.buy_price
                timediff = self.segments_list[self.segm_i][self.current_step, -1, self.timediff_col_ind]
                bars_distance_scaling = np.clip(timediff, 0.2, 2)  # (0.2 , 1.5, 2.28, 0.8,) ** 2

                quick_sell_penalty = -3 / self.idle_counter / bars_distance_scaling

                rew = gain * 4000 - self.reward_action_cost
                # rew = np.clip(rew, -3, 3)
                # print(reward, quick_sell_penalty)
                reward = rew + quick_sell_penalty
                # print(reward, rew, quick_sell_penalty)

                self.state = 0
                self.idle_counter = 0
                self.action_counter += 1

                self.cash = self.cash + price - self.score_action_cost
                self.score = self.cash
                was_valid = True
            else:
                reward = -10
                was_valid = False

        # price = self.df['price'].iloc[self.current_step]
        # reward = price * 10

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done and self.action_counter < 10:
            reward = -20

        elif done and self.state == 1:
            reward = -3
        # elif done:
        #     reward = 0

        reward -= self.idle_counter / 300

        if was_valid:
            reward = np.clip(reward, -9, 9)
        else:
            reward = np.clip(reward, -10, 10)
        # if done:
        #     return np.array([price]), reward, done, {}

        # next_price = self.df['price'].iloc[self.current_step]
        next_observation = self.observation

        return next_observation, reward, done, {}

    def evaluate(self, model, new_figure=True, segm_i=None, allow_plot=True):
        action_cost = self.score_action_cost

        ROWS = 3
        COLS = 1
        if new_figure and allow_plot:
            plt.subplots(ROWS, COLS, figsize=(25, 13), height_ratios=[5, 2, 3], dpi=150, sharex=True)
        timeoffset_x = segments_timestamps[self.segm_i][0, 0]
        # timeoffset_x = self.segments_list[self.segm_i][0, 0, self.timediff_col_ind]

        # price_x = self.segments_list[self.segm_i][:, -1, -1] - timeoffset_x
        price_x = segments_timestamps[self.segm_i][:, -1] - timeoffset_x
        price_y = self.segments_list[self.segm_i][:, -1, price_col]
        # plt.subplot(3, 1, 1)
        # plt.plot(price_x, price_y, color='black', alpha=0.4)
        # plt.subplot(3, 1, 2)
        # plt.plot(price_x, price_y, color='black', alpha=0.4)
        # plt.subplot(3, 1, 3)
        endgain_dict = {1: 0, 2: 0}

        for plt_i, det, det_state in [
                (1, False, None),  # (2, True, None),
                (3, True, False), (3, True, True)
        ]:
            if allow_plot:
                plt.subplot(ROWS, COLS, plt_i)
                plt.plot(price_x, price_y, color='black', alpha=0.4)

            # value = price_y[0]
            # cash = price_y[0]
            value_hist = []

            # state = 0
            green_x = []
            green_y = []
            red_x = []
            red_y = []
            invalid_red_x = []
            invalid_red_y = []
            invalid_green_x = []
            invalid_green_y = []
            reward_hist = []

            self.reset(segm_i=segm_i)

            done = False
            while not done:
                sample = self.segments_list[self.segm_i][self.current_step]

                step_price = sample[-1, price_col]
                xs = segments_timestamps[self.segm_i][self.current_step, -1] - timeoffset_x
                # xs = self.segments_list[seg_i][samp_i, -1] - timeoffset_x


                # arr = sample.ravel()
                # vec = np.concatenate([arr, [state]])
                # print()
                # print(f"{self.current_step}({done}): Getting obs")
                vec = self.observation

                # print(f"done:{done}")
                premove_state = env.state

                pred_act, _some = model.predict(vec, deterministic=det)
                obs, rew, done, _ = self.step(pred_act)
                reward_hist.append(rew)
                # print(f"done:{done}")
                # if np.random.random()<0.1:
                #     done=True
                # print(f"Ret: {ret}, some: {_some}")

                if det_state is True:
                    env.state = 1
                elif det_state is False:
                    env.state = 0

                if pred_act == 0:
                    # plt.scatter(xs, price, color='red')
                    if premove_state == 0:
                        green_x.append(xs)
                        green_y.append(step_price)

                    else:
                        invalid_green_x.append(xs)
                        invalid_green_y.append(step_price)

                elif pred_act == 2:
                    # plt.scatter(xs, price, color='green')
                    if premove_state == 1:
                        red_x.append(xs)
                        red_y.append(step_price)
                    else:
                        invalid_red_x.append(xs)
                        invalid_red_y.append(step_price)

                value = env.score
                if det_state is None:
                    value_hist.append(value)

            if allow_plot:
                plt.scatter(green_x, green_y, color='green', s=50)
                plt.scatter(red_x, red_y, color='red', s=50)
                plt.scatter(invalid_red_x, invalid_red_y, marker="x", color=(0.5, 0, 0), s=25)
                plt.scatter(invalid_green_x, invalid_green_y, marker="x", color=(0, 0.4, 0), s=25)
                if plt_i == 1:
                    plt.plot(price_x[:-1], value_hist, color='blue', alpha=0.5)
                    plt.subplot(ROWS, COLS, 2)
                    plt.plot(price_x[:-1], reward_hist, color='blue', alpha=0.5)

            # gain = value - price_y[0]
            gain = self.score - self.start_cash
            endgain_dict[plt_i] = gain

        if allow_plot:
            plt.subplot(ROWS, COLS, 1)
            plt.title(f"Gra, Buy: Green, Sell: Red. Endgain: {endgain_dict[1]:>4.4f}")
            plt.subplot(ROWS, COLS, 2)
            plt.title(f"Rewards")
            plt.subplot(ROWS, COLS, 3)
            plt.title("Podgląd miejsc kupna i sprzedaży")
            # plt.subplot(ROWS, COLS, 4)
            # plt.title("Podgląd miejsc sprzedaży")

            plt.suptitle(f"Action cost: {action_cost}")
            plt.tight_layout()

        return endgain_dict


def plot_gain(folder_path, color_pallete=None):
    file_path = os.path.join(folder_path, "gains.csv")
    model_name = os.path.basename(folder_path)
    print(f"Model: {model_name}")
    if not os.path.isfile(file_path):
        return "No file to plot"

    print(f"Plotting: {folder_path}")
    df = pd.read_csv(file_path)
    # print(df)
    mask1 = df["session"] == 0
    mask2 = df["segment_i"] == 0
    mask = mask1 & mask2
    args = np.argwhere(mask).ravel()
    args = np.concatenate([args, [len(df)]])
    # print(f"Args: {args}")
    # color = np.random.random()
    # print(COLORS[:5])

    offset_x = 0
    plt.figure(figsize=(20, 10), dpi=150)
    for start, stop in zip(args, args[1:]):
        print(f"Ploting: {start}:{stop} ({len(df)})")
        d_x = df.loc[start:stop - 1, "session"].to_numpy()
        d_y = df.loc[start:stop - 1, "segment_gain"]
        seg_i = df.loc[start:stop - 1, "segment_i"].to_numpy()

        d_x += offset_x
        offset_x = d_x[-1] + 1

        if color_pallete is not None:
            colors = color_pallete[seg_i]
        else:
            colors = None

        plt.scatter(d_x, d_y, c=colors)

    plt.title(f"{model_name}")
    plt.ylabel("Gain")
    plt.xlabel("Sesja")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "..", f"{model_name}.png"))


def start_stream_predict(
        baseline_model,
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
    # print(f"TIME FTS: {time_feats}")
    # print(f"TIME Window: {time_window}")

    state = 0
    last_segment_end = 0
    last_size = os.path.getsize(input_filepath)
    loaded_df_bars = pd.read_csv(input_filepath)
    curr_df_size = len(loaded_df_bars)
    if output_filepath is None:
        output_filepath = os.path.join(
                os.path.dirname(input_filepath),
                f"predicting_{arch_num:>03}_{lr}_{loss}.txt"
        )

    was_file = os.path.isfile(output_filepath)

    print("READY FOR NEW SAMPLES")
    # print(f"Model: {naming_ob.path}")
    print(f"Output file: {output_filepath}")

    with open(output_filepath, "at", buffering=1) as fp:
        if not was_file:
            input_file_columns = loaded_df_bars.columns
            ct = ",".join(cl for cl in input_file_columns)
            fp.write(f"{ct}\n")

        while True:
            time.sleep(0.1)
            size = os.path.getsize(input_filepath)
            if last_size == size:
                continue

            last_size = size
            prev_df_size = curr_df_size

            loaded_df_bars = pd.read_csv(input_filepath)
            # print(f"Shape: {loadead_df.shape}")
            curr_df_size = len(loaded_df_bars)
            missing_predictions = curr_df_size - prev_df_size

            # last_bar_ind = len(loadead_df) - 1
            print(f"File has changed by {missing_predictions} entries")
            loaded_segments, columns = preprocess_pipe_bars(
                    input_filepath,
                    get_n_bars=time_window,
                    clip_df_left=-1 * (missing_predictions + time_window * 3 + 5),
                    include_timestamp=False,
                    normalize_timediff=True,

                    workers=1,
                    # include_t

            )
            # print(f"Clip L: {-1 * (missing_predictions + time_window + 1)}")

            if len(loaded_segments) < 1:
                print(
                        f"{curr_df_size} RESET: Skipping update. No segments")
                # row = out_df.iloc[i]
                # print(loaded_segments)
                ser = loaded_df_bars.iloc[-1, :]
                fp.write(','.join(map(str, ser)))
                fp.write(",-1")
                fp.write("\n")
                state = 0
                continue

            # print(f"Segments: {loaded_segments}")
            loaded_segment = loaded_segments[-1]
            del loaded_segments

            if len(loaded_segment) < 1:
                print(
                        f"{curr_df_size} RESET: Skipping update. Too few bars: {loaded_segment.shape}")
                # row = out_df.iloc[i]
                ser = loaded_df_bars.iloc[-1, :]
                fp.write(','.join(map(str, ser)))
                fp.write(",-1")
                fp.write("\n")
                state = 0
                continue

            # print(f"Missing predictions: {missing_predictions}")

            for i in range(missing_predictions):
                clipr = (-1 * missing_predictions + i + 1)
                if clipr != 0:
                    sequences_3d = loaded_segment[:clipr].copy()  # BARS ONLY
                    cur_raw_df = loaded_df_bars[:clipr]
                else:
                    sequences_3d = loaded_segment.copy()
                    cur_raw_df = loaded_df_bars
                # print(f"Seq3d shape: {sequences_3d.shape}")

                # print(f"Predicting from: {last_segment_end}:{last_bar_ind + i + 2}")
                t0 = time.time()

                # "CLEAN"
                # # dataframe = preprocess(dataframe, first_sample_date=None) # UNIFORM
                # if len(dataframe) <= 1:
                #     print(
                #             f"{last_bar_ind + i + 1} RESET: Skipping iteration: {i}. Df too short: {dataframe.shape}")
                #     # row = out_df.iloc[i]
                #     ser = loadead_df.iloc[last_bar_ind + 1 + i]
                #     fp.write(','.join(map(str, ser)))
                #     fp.write(",-1")
                #     fp.write("\n")
                #     state = 0
                #
                #     if was_ok:
                #         last_segment_end = last_bar_ind + 1 + i
                #         was_ok = False
                #     continue

                # segments, columns = generate_interpolated_data(
                #         dataframe=dataframe, include_time=False,
                #         interval_s=interval_s, split_interval_s=split_interval_s
                # )
                # print(f"Splitted into {len(segments)} segments")

                # list_ofsequences = [to_sequences_forward(arr, 10, [1])[0] for arr in segments]
                # current_sequence, _ = to_sequences_forward(segments[-1], 10, [1])

                # print(f"current sequence: {current_sequence.shape}")

                "UNIFORM"
                # if len(current_sequence) <= 0:
                #     print(f"{last_bar_ind + i + 1} RESET: Skipping iteration: {i} too short sequence")
                #     ser = loadead_df.iloc[last_bar_ind + 1 + i]
                #     # ser['act'] = -1
                #     fp.write(','.join(map(str, ser)))
                #     fp.write(",-1")
                #     fp.write("\n")
                #     state = 0
                #     if was_ok:
                #         last_segment_end = last_bar_ind + 1 + i
                #         was_ok = False
                #     continue

                "BARS"
                if len(sequences_3d) <= 0:
                    print(f"{prev_df_size + i + 1} RESET: Skipping iteration: {i} too few bars")
                    ser = cur_raw_df.iloc[-1]
                    # ser['act'] = -1
                    fp.write(','.join(map(str, ser)))
                    fp.write(",-1")
                    fp.write("\n")
                    state = 0
                    continue

                # print(f"Predicting from sequences: {current_sequence.shape}")

                pred_state = np.array(state).reshape(1, 1, 1)
                # pred_arr = current_sequence[-1, :, :][np.newaxis, :, :]
                # pred_arr = sequences_3d[-1, :, :].reshape(1, time_window, time_feats)
                pred_arr = sequences_3d[-1, :, :]
                # print(f"pred Arr shape: {pred_arr.shape}")

                # print(f"Pred state: {pred_state}, {pred_state.shape}")
                # print(f"Predict shapes: {pred_arr.shape}, {pred_state.shape}")
                # predicted = model_keras.predict([pred_arr, pred_state], verbose=False)
                # predicted = model_keras.predict([pred_arr, pred_state], verbose=False)
                # print(f"Predict: {pred_arr}")
                # print(f"Pred state: {pred_state}")

                obs = np.concatenate([pred_arr.ravel(), pred_state.ravel()])
                # print(f"obs: {obs.shape}")

                act, _ = baseline_model.predict(obs)
                # act = np.argmax(predicted, axis=1)[0]
                print(f"Predicted: {act}")
                # print(f"Act: {act} from state: {state}")

                ser = cur_raw_df.iloc[-1]
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
                print(prev_df_size + i + 1, i,
                      f"Loop duration: {loop_dur:>5.2}s",
                      f"Act: {act}, End state:{state}, was state: {prev_state}")
                # was_ok = True

            print("========")


# Creating a sample DataFrame
# data = {'price': [10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 22.0, 19.0, 16.0, 14.0]}
# df = pd.DataFrame(data)

if __name__ == "__main__":
    files = ["obv_600.txt"]
    # files = ["on_balance_volume.txt"]
    file_path = path_data_folder + files[0]
    time_size = 80

    "ARG PARSING"
    "=Flags="
    print("Argumenty wywołania:", exec_arguments)
    evalonly = exec_arguments.evalonly
    noeval = exec_arguments.noeval
    skip_first_plot = exec_arguments.skip
    plot_gains_only = exec_arguments.plot
    allow_plot = not exec_arguments.noplot
    # print(f"Allow plot: {allow_plot}")

    model_type = exec_arguments.modeltype
    if model_type is None:
        model_type = "ppo"
    else:
        model_type = model_type.lower()

    model_num = exec_arguments.modelnum
    if model_num is None:
        model_num = 1

    reward_fnum = exec_arguments.reward
    if reward_fnum is None:
        reward_fnum = 1

    run_live = exec_arguments.live
    path_input = exec_arguments.pathinput
    path_output = exec_arguments.pathoutput

    "=============="
    model_textparams = f"mt{model_type}-mn{model_num}-r{reward_fnum}"
    path_modelfolder_baseline = os.path.join(path_models, "baseline", model_textparams, "")

    if plot_gains_only:
        COLORS = list(CSS4_COLORS.keys())
        # random.shuffle(COLORS)
        COLORS = np.array(COLORS)
        # print(f"COLORS: {len(COLORS)}")
        # print(COLORS)
        # for i, k in enumerate(COLORS):
        #     print(i, k)

        COLORS = COLORS[
            [
                    44,  # BLue
                    105,  # Orange
                    21,  # Dark blue
                    54,  # Green
                    82,  # Lime
                    22,  # Dark cyan
                    115,  # Pink
                    118,  # Purple
                    138,  # Teal
                    32,  # Dark red
                    10,  # Blue violet
                    131,  # Slate Blue
                    19,  # Crimson
                    16,  # Coral
                    60,  # indigo
                    103,  # olive
                    138,  # Teal


            ],
        ]

        # COLORS = COLORS[1::12]
        print(COLORS)
        print(f"Got Colors size: {len(COLORS)}")

        use("ggplot")
        folders = glob.glob(f"{path_models_baseline}*")
        # print(f"found folders: {folders}")
        folders = [fold for fold in folders if os.path.isdir(fold)]
        # print(f"found folders: {folders}")
        for folder_ph in folders:
            plot_gain(folder_ph, color_pallete=COLORS.copy())
        sys.exit(0)

    lclip = None

    if run_live:
        print(f"Running model {model_textparams} live.")
        # time.sleep(4)
        rclip = 500

    else:
        rclip = 5000
        print(f"Starting model: {model_textparams}\n" * 5)

        # time.sleep(4)
    os.makedirs(path_modelfolder_baseline, exist_ok=True)

    gain_fp = path_modelfolder_baseline + "gains.csv"
    if not os.path.isfile(gain_fp):
        with open(gain_fp, "wt") as fh:
            fh.write("session,segment_i,segment_gain\n")

    trainsegments_ofsequences3d, columns = preprocess_pipe_bars(
            file_path, get_n_bars=time_size,
            add_timediff_feature=True,
            include_timestamp=True,
            normalize_timediff=True,
            minsamples_insegment=300,
            clip_df_left=lclip,
            clip_df_right=rclip,
            # first_sample_date="2023-6-29",  # only for on_balance_volume
    )

    # print(columns[0])
    price_col = np.argwhere(columns[0] == 'last').ravel()[0]
    timestamp_col = np.argwhere(columns[0] == 'timestamp_s').ravel()[0]

    if price_col > timestamp_col:
        "Compensate later index"
        price_col -= 1

    segments_timestamps = [segm[:, :, timestamp_col] for segm in trainsegments_ofsequences3d]
    columns = np.delete(columns[0], timestamp_col)
    trainsegments_ofsequences3d = [
            np.delete(segm, timestamp_col, axis=2) for segm in
            trainsegments_ofsequences3d
    ]

    env = TradingEnvironment(trainsegments_ofsequences3d, columns, reward_func=reward_fnum)
    env.reset()

    use("ggplot")

    lr = 5e-6
    ent_coef = 1e-3
    arch_nodes = 2000
    batch_size = 3000

    model_ph = path_modelfolder_baseline + "model.bs3"

    if model_type == "dqn":
        if os.path.isfile(model_ph):
            print("Creating DQN")
            model = DQN.load(
                    model_ph,
                    env, verbose=1,
                    learning_rate=lr,
                    policy_kwargs=dict(net_arch=[arch_nodes, arch_nodes]),
                    batch_size=batch_size,
            )
        else:
            print("Loading DQN")
            skip_first_plot = True
            model = DQN(
                    'MlpPolicy', env, verbose=1,
                    learning_rate=lr,
                    policy_kwargs=dict(net_arch=[arch_nodes, arch_nodes]),
                    batch_size=batch_size,
            )

    elif model_type == 'ppo':
        if os.path.isfile(model_ph):
            print(f"Loading PPO: {model_ph}")
            model = PPO.load(
                    model_ph, env=env,
                    learning_rate=lr,
                    ent_coef=ent_coef,
                    batch_size=batch_size,
                    clip_range=0.05,
            )
        else:
            skip_first_plot = True
            print("Creating PPO")
            model = PPO(
                    'MlpPolicy', env, verbose=1,
                    learning_rate=lr,
                    ent_coef=ent_coef,
                    # batch_size=300,
                    batch_size=batch_size,
                    policy_kwargs=dict(net_arch=[arch_nodes, arch_nodes]),
                    clip_range=0.05,
            )
    else:
        raise ValueError(f"Wrong model type: {model_type}")

    print("POLICY:")
    for name, param in model.policy.named_parameters():
        print(name, param.shape)

    print(model.learning_rate)
    print(model.policy.optimizer)

    # model.policy.optimizer.param_groups[0]['lr'] = 1e-3
    # # model.set_parameters(dict(lr=0.1))
    #
    # print(model.policy.optimizer)
    # print(model.learning_rate)
    if run_live:
        pass
        print("SUCCES LOADING")
        start_stream_predict(
                model,
                input_filepath=path_input, output_filepath=path_output,
                time_window=time_size,
        )
    else:
        if evalonly:
            games = 1
        else:
            games = 100

        for session in range(games):
            if (session != 0 or skip_first_plot) and not evalonly:
                model.learn(total_timesteps=25_000)
                model.save(model_ph)

            with open(gain_fp, "at") as fh:
                for seg_i, segment in enumerate(trainsegments_ofsequences3d):
                    env.reset(seg_i)
                    endgain_dict = env.evaluate(model, segm_i=seg_i, allow_plot=allow_plot)
                    endgain = endgain_dict[1]
                    # print("WRITING:", endgain)
                    fh.write(f"{session},{seg_i},{endgain:>4.5f}\n")

                    if allow_plot:
                        plt.savefig(
                                path_modelfolder_baseline + f"{model_type}-seg-{seg_i}-({session}).png")
                        print(f"Saved plot({session}): {model_type}-eval_seg-{seg_i}.png")
                        plt.close()

            print(f"Session Ended: {session}")
