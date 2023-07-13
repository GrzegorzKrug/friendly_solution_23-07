import numpy as np


def initialize_agents(agents_n, start_stock=0):
    """

    Args:
        agents_n:
        start_stock:

    Returns:
        Discrete state: Integer of assets
        Hidden state: [Cash, Starting cash]

    """
    # states = np.random.random((agents_n, 1)) * 2 + 1
    # discrete_state = np.tile(start_stock, (agents_n, 1))

    # state = np.concatenate([agents_arr, cash], axis=1)
    # starting_money = cash.copy()
    # staring_state = state.copy()
    # return state, staring_state
    # states = np.random.random((agents_n, 1)) + 0.5
    disc_state = np.zeros((agents_n, 1))
    cargo = np.zeros((agents_n, 1), dtype=int) + start_stock
    cash = np.random.random((agents_n, 1)) + 1

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
    for i, (dsc_state, hid_state, act) in enumerate(zip(discrete_states, hidden_states, actions)):
        pass
    return discrete_states, hidden_states
