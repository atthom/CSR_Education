import numpy as np
from scipy.signal.windows import blackman

def reward(state: dict) -> float:
    """
    Compute the reward for the current state.
    @param state: the current state.
    @return: the reward for the current state.
    """
    # The result of print instructions will be displayed in the console
    print('Current state is:')
    #print(state)
    print(state['in_v_21x16'].shape)
    gray_pixel_line = state['in_v_21x16'][-1, :, 0]
    window = blackman(21)
    cost = 1 - gray_pixel_line
    cost = cost * window
    return sum(cost)
