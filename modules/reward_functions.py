import numpy as np
import numba
import typing


INVALID_MOVE = -10
SMALL_MISTAKE = INVALID_MOVE * 0.02
# SMALL_MISTAKE = -0.5

ENDING_GAIN_SCALE = 20


# ROI_IMORTANCE = 10
# LOW_PRICE_EPS = 0.05

# fullday_seconds = 24 * 60 * 60
# timeout_5min = 5 * 60 / fullday_seconds
# timeout_10min = 10 * 60 / fullday_seconds
# timeout_15min = 15 * 60 / fullday_seconds
# timeout_20min = 20 * 60 / fullday_seconds


class RewardStore:
    funcs = dict()

    @classmethod
    def add(cls, key):
        if key in cls.funcs:
            raise KeyError(f"There is function assigned to key: {key}")

        def wrapper(func):
            "Store func, return Func"
            cls.funcs[key] = func
            return func

        return wrapper

    @classmethod
    def get(cls, key):
        f = cls.funcs.get(key, None)
        if f is None:
            raise KeyError(f"There is no key in store: {key}")
        return f


@numba.njit()
def reward_fun_template(
        env_arr, actor_state_arr, action: int,
        hidden_arr,
        done=False,
) -> (float, bool):
    """

    Returns:
        - Score: Float
        - Valid: bool
    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.
    """


@RewardStore.add(1)
# @numba.njit()
def reward_fun_1(
        env_arr, discrete_state, action: int,
        hidden_arr,
        done=False,
        price=0,
        # price_col_ind=0,
        # initial_cash=0,
):
    """Assign Reward"""

    # cash, cargo, last_sell, last_buy, last_transaction_price = wallet

    # time_now = last_row[0]
    # price_sample = last_row[4]
    cash, initial_cash, discrete_stock = hidden_arr

    # price_now = env_arr[0, price_col_ind]
    # print(f"Price now:", price_now)
    if done:
        # cash = discrete_state[0]
        end_cash = cash + price * discrete_stock
        gain = end_cash - initial_cash
        return gain * ENDING_GAIN_SCALE, True
    else:
        if action == 0:
            "BUY"
            pass
        elif action == 1:
            "PASS"
            pass
        elif action == 2:
            "SELL"
            if discrete_stock <= 0:
                return INVALID_MOVE, False

        return 0, True


if __name__ == "__main__":
    walet = np.zeros((1, 2))
    env_arr = np.zeros((10, 20))
    reward_fun_1(env_arr, walet, 2)
    # reward_fun_1()
    # np.max
