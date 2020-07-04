import os
import math
import logging
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta



def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(close_data, volume_data, date_data, economy_data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1
    close_block = close_data[d: t + 1] if d >= 0 else -d * [close_data[0]] + close_data[0: t + 1]  # pad with t0
    volume_block = volume_data[d: t+1] if d >= 0 else -d * [volume_data[0]] + volume_data[0: t + 1]

    date = date_data[t]
    prevtime = date - relativedelta(months=3)
    prev_prevtime = prevtime - relativedelta(months=3)
    prevtime = str(prevtime.date().replace(day=1))
    prev_previtme = str(prev_prevtime.date().replace(day=1))

    economy_block = economy_data[prev_previtme:prevtime]

    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(close_block[i + 1] - close_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(volume_block[i+1] - volume_block[i]))
    for column in economy_block.columns:
        for i in range(len(economy_block)-1):
            res.append(sigmoid(economy_block[column][i+1] - economy_block[column][i]))

    return np.array([res])
