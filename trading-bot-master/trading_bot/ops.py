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


def get_state(date, close, volume, open, high, low, kdj_k, kdj_d, kdj_j, macd, macds, macdo, cci, rsi, avg5, avg10, avg20, t, n_days):
    """Returns an n-day state representation ending at time t
    """

    d = t - n_days + 1
    open_block = open[d:t+1] if d>=0 else -d * [open[0]] + open[0:t+1]
    close_block = close[d: t+1] if d >= 0 else -d * [close[0]] + close[0: t + 1]  # pad with t0
    high_block = high[d:t+1] if d>=0 else -d * [high[0]] + high[0:t+1]
    low_block = low[d:t+1] if d>=0 else -d * [low[0]] + low[0:t+1]
    volume_block = volume[d: t+1] if d >= 0 else -d * [volume[0]] + volume[0: t + 1]
    kdj_k_block = kdj_k[d: t+1] if d >= 0 else -d * [kdj_k[0]] + kdj_k[0: t+1]
    kdj_d_block = kdj_d[d: t+1] if d >=0 else -d * [kdj_d[0]] + kdj_d[0: t+1]
    kdj_j_block = kdj_j[d: t+1] if d >= 0 else -d * [kdj_j[0]] + kdj_j[0: t+1]
    macd_block = macd[d: t+1] if d >= 0 else -d * [macd[0]] + macd[0: t+1]
    macds_block = macds[d: t + 1] if d >= 0 else -d * [macds[0]] + macds[0: t + 1]
    macdo_block = macdo[d: t + 1] if d >= 0 else -d * [macdo[0]] + macdo[0: t + 1]
    cci_block = cci[d: t+1] if d >= 0 else -d * [cci[0]] + cci[0: t + 1]
    rsi_block = rsi[d: t+1] if d >=0 else -d * [rsi[0]] + rsi[0: t+1]
    avg5_block = avg5[d: t+1] if d>=0 else -d * [avg5[0]] + avg5[0: t+1]
    avg10_block = avg10[d: t + 1] if d >= 0 else -d * [avg10[0]] + avg10[0: t + 1]
    avg20_block = avg20[d: t + 1] if d >= 0 else -d * [avg20[0]] + avg20[0: t + 1]

    res = []
    '''
    for i in range(n_days - 1):
        res.append((avg5_block[i + 1] - avg5_block[i]))
    for i in range(n_days - 1):
         res.append((avg10_block[i+1] - avg10_block[i]))
    '''
    for i in range(n_days - 1):
        res.append((open_block[i + 1] - open_block[i]))
    for i in range(n_days - 1):
        res.append((high_block[i + 1] - high_block[i]))
    for i in range(n_days - 1):
        res.append((low_block[i + 1] - low_block[i]))
    for i in range(n_days - 1):
        res.append((close_block[i + 1] - close_block[i]))
    for i in range(n_days - 1):
        res.append((avg20_block[i + 1] - avg20_block[i]))
    for i in range(n_days - 1):
        res.append((volume_block[i + 1] - volume_block[i]))
    for i in range(n_days-1):
        res.append(kdj_k_block[i+1])
    for i in range(n_days-1):
        res.append(kdj_d_block[i+1])
    for i in range(n_days-1):
        res.append(kdj_j_block[i+1])
    for i in range(n_days-1):
        res.append(macd_block[i+1])
    for i in range(n_days-1):
        res.append(macds_block[i+1])
    for i in range(n_days-1):
        res.append(macdo_block[i+1])
    for i in range(n_days-1):
        res.append(cci_block[i+1])
    for i in range(n_days-1):
        res.append(rsi_block[i+1])


    return np.array([res])
