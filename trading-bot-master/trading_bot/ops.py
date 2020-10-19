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


def get_state(open_data, high_data, low_data, close_data, volume_data, date_data, economy_data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1
    pre_5d = t - n_days - 4
    pre_10d = t - n_days - 9
    pre_20d = t - n_days - 19

    open_block = open_data[d: t+1] if d >= 0 else -d * [open_data[0]] + open_data[0: t+1]
    high_block = high_data[d: t+1] if d >= 0 else -d * [high_data[0]] + high_data[0: t+1]
    low_block = low_data[d: t+1] if d >= 0 else -d * [low_data[0]] + low_data[0: t+1]
    close_block = close_data[d: t+1] if d >= 0 else -d * [close_data[0]] + close_data[0: t + 1]  # pad with t0
    volume_block = volume_data[d: t+1] if d >= 0 else -d * [volume_data[0]] + volume_data[0: t + 1]

    # 5일선을 구하기 위해 사용
    pre_5d_block = close_data[pre_5d: t+1] if pre_5d >= 0 else -pre_5d * [close_data[0]] + close_data[0: t+1]
    # 10일선을 구하기 위해 사용
    pre_10d_block = close_data[pre_10d: t+1] if pre_10d >= 0 else -pre_10d * [close_data[0]] + close_data[0: t+1]
    # 20일선을 구하기 위해 사용
    pre_20d_block = close_data[pre_20d: t + 1] if pre_20d >= 0 else -pre_20d * [close_data[0]] + close_data[0: t + 1]

    avg_5d_block = []
    for i in range(n_days):
        avg_5d = sum(pre_5d_block[i: i+5])/5
        avg_5d_block.append(avg_5d)

    avg_10d_block = []
    for i in range(n_days):
        avg_10d = sum(pre_10d_block[i: i+10])/10
        avg_10d_block.append(avg_10d)

    avg_20d_block = []
    for i in range(n_days):
        avg_20d = sum(pre_20d_block[i: i+30])/30
        avg_20d_block.append(avg_20d)

    '''
    date = date_data[t]
    prevtime = date - relativedelta(months=3)
    prev_prevtime = prevtime - relativedelta(months=3)
    prevtime = str(prevtime.date().replace(day=1))
    prev_previtme = str(prev_prevtime.date().replace(day=1))

    economy_block = economy_data[prev_previtme:prevtime]
    '''

    # sigmoid 이용
    '''
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(open_block[i + 1] - open_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(high_block[i + 1] - high_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(low_block[i + 1] - low_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(close_block[i + 1] - close_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(avg_5d_block[i + 1] - avg_5d_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(avg_10d_block[i + 1] - avg_10d_block[i]))
    for i in range(n_days - 1):
        res.append(sigmoid(avg_20d_block[i + 1] - avg_20d_block[i]))

    for i in range(n_days - 1):
        res.append(sigmoid(volume_block[i+1] - volume_block[i]))
    '''
    '''
    for column in economy_block.columns:
        for i in range(len(economy_block)-1):
            res.append(sigmoid(economy_block[column][i+1] - economy_block[column][i]))
    '''

    # raw값 그대로 이용
    res = []
    for i in range(n_days - 1):
        res.append(open_block[i])
    for i in range(n_days - 1):
        res.append(high_block[i])
    for i in range(n_days - 1):
        res.append(low_block[i])
    for i in range(n_days - 1):
        res.append(close_block[i])
    for i in range(n_days - 1):
        res.append(avg_5d_block[i])
    for i in range(n_days - 1):
        res.append(avg_10d_block[i])
    for i in range(n_days - 1):
        res.append(avg_20d_block[i])
    for i in range(n_days - 1):
        res.append(volume_block[i])

    '''
    res = np.zeros((7,10))
    res[0] = open_block
    res[1] = high_block
    res[2] = low_block
    res[3] = close_block
    res[4] = volume_block
    res[5] = avg_5d_block
    res[6] = avg_10d_block
    '''
    return np.array([res])
