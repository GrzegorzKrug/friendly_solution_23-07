import numpy as np


def initialize_agents(agents_n, start_stock=0):
    """

    Args:
        agents_n:
        start_stock:

    Returns:
        Discrete state: [bool cargo,]
        Hidden state: [Cash, starting cash, cargo, buy price]

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
    buy_price = cargo.astype(float)
    idle_counter = cash * 0
    hidden_state = np.concatenate([cash, cash.copy(), cargo, buy_price, idle_counter], axis=1)

    return disc_state, hidden_state


def resolve_actions_multibuy(cur_step_price, discrete_states, hidden_states, actions, price_mod=1):
    """

    Args:
        cur_step_price:
        discrete_states:
        hidden_states:
        price:
        price_mod:

    Returns:

    """
    raise NotImplemented("Fix last new hidden states ")

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


def resolve_actions_singlebuy(
        cur_step_price, discrete_states, hidden_states, actions, price_mod=1,
        action_cost=0.0):
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
    for ag_i, (dsc_state, hid_state, act) in enumerate(zip(discrete_states, hidden_states, actions)):
        if act == 0:
            "BUY"
            if new_disc_state[ag_i] == 0:
                "BUY only if have none"
                new_disc_state[ag_i] = 1
                new_hidden_state[ag_i][0] -= cur_step_price * price_mod + action_cost
                new_hidden_state[ag_i][2] += 1
                new_hidden_state[ag_i][3] = cur_step_price  # Remember buy price
                new_hidden_state[ag_i][4] = 0  # Reset idle counter

        elif act == 1:
            "IDLE"
            new_hidden_state[ag_i][4] += 1  # more idle

        elif act == 2:
            "SELL"

            new_disc_state[ag_i] = 0  # 0 Assets in state
            # new_hidden_state[ag_i][3] = 0  # Buy price 0

            if new_hidden_state[ag_i][2] <= 0:
                "Can not sell"
                new_hidden_state[ag_i][2] = 0  # Set 0 asset
            else:
                "Can sell"
                new_hidden_state[ag_i][0] += cur_step_price * price_mod - action_cost  # Gain cash wallet
                new_hidden_state[ag_i][2] -= 1  # Less cargo
                new_hidden_state[ag_i][4] = 0  # Reset idle counter

                # if new_hidden_state[i][2] > 0:
                #     new_disc_state[i] = 1
                # else:

    return new_disc_state, new_hidden_state
