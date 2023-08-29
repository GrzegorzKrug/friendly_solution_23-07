import os


module_path = './gym_anytrading'
if not os.path.exists(module_path):
    os.chdir('../')
# ------------------------------------------------------------

# import quantstats as qs

from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt

import gym
from gym import spaces
import numpy as np

from common_settings import path_data_folder, path_models
from preprocess_data import preprocess_pipe_bars

from matplotlib.style import use


class TradingEnvironment(gym.Env):
    def __init__(self, segments_list, columns):
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

    def reset(self):
        self.segm_i = np.random.randint(0, len(self.segments_list))
        self.current_step = 0
        self.max_steps = len(self.segments_list[self.segm_i]) - 1
        self.hist_actions = []
        self.state = 0
        self.action_counter = 0
        self.idle_counter = 0
        self.buy_price = 0
        print(f"RESETING: segm_i: {self.segm_i}")

        return self.observation

    @property
    def observation(self):
        vec = self.segments_list[self.segm_i][self.current_step].ravel()
        return np.concatenate([vec, [self.state]])

    def step(self, action):
        # print(f"Step: {action}")

        assert self.action_space.contains(action)
        price = self.segments_list[self.segm_i][self.current_step, -1, self.price_col_ind]
        # fut_price = self.segments_list[self.segm_i][self.current_step + 1, -1, self.price_col_ind]
        # price = time_arr[-1, self.price_col_ind]
        action_cost = 0.01
        self.idle_counter += 1

        if action == 0:
            if self.state == 0:
                reward = -price - action_cost
                self.buy_price = price
                self.state = 1

                self.action_counter += 1
                self.idle_counter = 0
            else:
                reward = -10


        elif action == 1:
            reward = 0
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

                rew = gain * 1000
                rew = np.clip(rew, -5, 5)
                # print(reward, quick_sell_penalty)
                reward = rew + quick_sell_penalty
                # print(reward, rew, quick_sell_penalty)

                self.state = 0
                self.idle_counter = 0
                self.action_counter += 1
            else:
                reward = -10

        # price = self.df['price'].iloc[self.current_step]
        # reward = price * 10

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done and self.action_counter < 10:
            reward = -20

        elif done and self.state == 1:
            reward = -1
        # elif done:
        #     reward = 0

        reward -= self.idle_counter / 100

        # if done:
        #     return np.array([price]), reward, done, {}

        # next_price = self.df['price'].iloc[self.current_step]
        next_observation = self.observation

        return next_observation, reward, done, {}

    def render(self):
        pass


# Creating a sample DataFrame
# data = {'price': [10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 22.0, 19.0, 16.0, 14.0]}
# df = pd.DataFrame(data)

if __name__ == "__main__":
    files = ["obv_600.txt"]
    # files = ["on_balance_volume.txt"]
    file_path = path_data_folder + files[0]
    time_size = 80

    path_baseline_models = os.path.join(path_models, "baseline", "")
    os.makedirs(path_baseline_models, exist_ok=True)

    trainsegments_ofsequences3d, columns = preprocess_pipe_bars(
            file_path, get_n_bars=time_size,
            add_timediff_feature=True,
            include_timestamp=True,
            normalize_timediff=True,
            minsamples_insegment=300,
            # clip_df_left=8000,
            clip_df_right=35000,
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

    env = TradingEnvironment(trainsegments_ofsequences3d, columns)
    env.reset()

    use("ggplot")

    model_type = "ppo"
    lr = 1e-5
    ent_coef = 1e-3
    arch_nodes = 2000
    batch_size = 500

    if model_type == "dqn":
        model = DQN(
                'MlpPolicy', env, verbose=1,
                learning_rate=1e-5,
                policy_kwargs=dict(net_arch=[arch_nodes, arch_nodes]),
                batch_size=300,
        )
        model_ph = path_baseline_models + "model1-dqn.bs3"

    elif model_type == 'ppo':
        model = PPO(
                'MlpPolicy', env, verbose=1,
                learning_rate=lr,
                ent_coef=ent_coef,
                # batch_size=300,
                batch_size=batch_size,
                policy_kwargs=dict(net_arch=[arch_nodes, arch_nodes]),
                clip_range=0.1,
        )
        model_ph = path_baseline_models + "model2-ppo.bs3"
    else:
        raise ValueError(f"Wrong model type: {model_type}")

    if os.path.isfile(model_ph):
        model = model.load(
                model_ph, env=env,
                learning_rate=lr,
                ent_coef=ent_coef,
                batch_size=batch_size,
                clip_range=0.1,
        )

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

    for session in range(100):
        if session != 0:
            model.learn(total_timesteps=50_000)
            model.save(model_ph)
        # print("MODEL SAVED")

        for seg_i, segment in enumerate(trainsegments_ofsequences3d):
            action_cost = 0.0001

            ROWS = 4
            COLS = 1
            plt.subplots(ROWS, COLS, figsize=(25, 13), height_ratios=[5, 2, 1, 1], dpi=150, sharex=True)
            timeoffset_x = segments_timestamps[seg_i][0, 0]

            price_x = segments_timestamps[seg_i][:, -1] - timeoffset_x
            price_y = trainsegments_ofsequences3d[seg_i][:, -1, price_col]
            # plt.subplot(3, 1, 1)
            # plt.plot(price_x, price_y, color='black', alpha=0.4)
            # plt.subplot(3, 1, 2)
            # plt.plot(price_x, price_y, color='black', alpha=0.4)
            # plt.subplot(3, 1, 3)
            endgain = {1: 0, 2: 0}

            for plt_i, det, det_state in [
                    (1, False, None), (2, True, None),
                    (3, True, False), (4, True, True)
            ]:
                plt.subplot(ROWS, COLS, plt_i)
                plt.plot(price_x, price_y, color='black', alpha=0.4)

                value = price_y[0]
                cash = price_y[0]
                value_hist = []

                state = 0
                green_x = []
                green_y = []
                red_x = []
                red_y = []
                invalid_red_x = []
                invalid_red_y = []
                invalid_green_x = []
                invalid_green_y = []

                for samp_i, sample in enumerate(segment):
                    step_price = sample[-1, price_col]
                    # xs = sample[-1, timestamp_col] - timeoffset_x
                    xs = segments_timestamps[seg_i][samp_i, -1] - timeoffset_x

                    arr = sample.ravel()
                    vec = np.concatenate([arr, [state]])

                    ret, _some = model.predict(vec, deterministic=det)
                    # print(f"Ret: {ret}, some: {_some}")
                    if det_state is True:
                        state = 1
                    elif det_state is False:
                        state = 0

                    if ret == 0:
                        # plt.scatter(xs, price, color='red')
                        if state == 0:
                            green_x.append(xs)
                            green_y.append(step_price)
                            value -= action_cost
                            cash = cash - step_price - action_cost

                        else:
                            invalid_green_x.append(xs)
                            invalid_green_y.append(step_price)

                        state = 1
                    elif ret == 2:
                        # plt.scatter(xs, price, color='green')
                        if state == 1:
                            red_x.append(xs)
                            red_y.append(step_price)
                            cash = cash + step_price - action_cost
                            value = cash
                        else:
                            invalid_red_x.append(xs)
                            invalid_red_y.append(step_price)
                        state = 0

                    if det_state is None:
                        value_hist.append(value)

                plt.scatter(green_x, green_y, color='green', s=50)
                plt.scatter(red_x, red_y, color='red', s=50)
                plt.scatter(invalid_red_x, invalid_red_y, marker="x", color=(0.5, 0, 0), s=25)
                plt.scatter(invalid_green_x, invalid_green_y, marker="x", color=(0, 0.4, 0), s=25)
                if det_state is None:
                    plt.plot(price_x, value_hist, color='blue', alpha=0.5)

                gain = value - price_y[0]
                endgain[plt_i] = gain

            plt.subplot(ROWS, COLS, 1)
            plt.title(f"Gra, Buy: Green, Sell: Red. Endgain: {endgain[1]:>4.4f}")
            plt.subplot(ROWS, COLS, 2)
            plt.title(f"Deterministyczna gra, Buy: Green, Sell: Red. Endgain: {endgain[2]:>4.4f}")
            plt.subplot(ROWS, COLS, 3)
            plt.title("Podgląd miejsc kupna")
            plt.subplot(ROWS, COLS, 4)
            plt.title("Podgląd miejsc sprzedaży")

            plt.suptitle(f"Action cost: {action_cost}")
            plt.tight_layout()
            plt.savefig(path_baseline_models + f"{model_type}-seg-{seg_i}-({session}).png")
            # print(f"Saved plot: {model_type}-eval_seg-{seg_i}.png")
            plt.close()

        print(f"PLOT Session Ended: {session}")
