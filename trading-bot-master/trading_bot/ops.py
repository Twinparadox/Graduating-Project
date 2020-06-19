import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(close_data, volume_data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1
    close_block = close_data[d: t + 1] if d >= 0 else -d * [close_data[0]] + close_data[0: t + 1]  # pad with t0
    volume_block = volume_data[d: t+1] if d >= 0 else -d * [volume_data[0]] + volume_data[0: t + 1]
    #return np.array([block])
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(close_block[i + 1] - close_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(volume_block[i+1] - volume_block[i]))
    return np.array([res])
