import numpy as np


def initialize_agents(agents_n, start_stock=0):
    """

    Args:
        agents_n:
        start_stock:

    Returns:
        Discrete state: Integer of assets
        Hidden state: [Cash, Starting cash, cargo]

    """
    # states = np.random.random((agents_n, 1)) * 2 + 1
    # discrete_state = np.tile(start_stock, (agents_n, 1))

    # state = np.concatenate([agents_arr, cash], axis=1)
    # starting_money = cash.copy()
    # staring_state = state.copy()
    # return state, staring_state
    # states = np.random.random((agents_n, 1)) + 0.5
    disc_state = np.zeros((agents_n, 1))

    cash = np.random.random((agents_n, 1)) + 1
    cargo = np.zeros((agents_n, 1), dtype=int) + start_stock
    hidden_state = np.concatenate([cash, cash.copy(), cargo], axis=1)

    return disc_state, hidden_state


def resolve_actions(cur_step_price, discrete_states, hidden_states, actions, price_mod=1):
    """

    Args:
        cur_step_price:
        discrete_states:
        hidden_states:
        price:
        price_mod:

    Returns:

    """
    new_disc_state = discrete_states.copy()
    new_hidden_state = hidden_states.copy()
    for i, (dsc_state, hid_state, act) in enumerate(zip(discrete_states, hidden_states, actions)):
        if act == 0:
            "BUY"
            new_disc_state[i] = 1
            new_hidden_state[i][0] -= cur_step_price * price_mod
            new_hidden_state[i][2] += 1

        elif act == 1:
            pass
        elif act == 2:
            "SELL"
            if new_hidden_state[i][2] <= 0:
                new_hidden_state[i][2] = 0
            else:
                new_hidden_state[i][2] -= 1
                new_hidden_state[i][0] += cur_step_price * price_mod

                if new_hidden_state[i][2] > 0:
                    new_disc_state[i] = 1
                else:
                    new_disc_state[i] = 0

    return new_disc_state, new_hidden_state
