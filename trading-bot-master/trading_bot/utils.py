import os
import math
import logging

import pandas as pd
import numpy as np

import keras.backend as K


# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))



def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS asset:{} Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), format_position(result[3]), result[4]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {} asset:{} Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), format_position(result[3]), result[4]))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    print('get_stock_data()')
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m")
    return list(df['date']), list(df['close']), list(df['volume']), \
           list(df['open']), list(df['high']), list(df['low']), \
           list(df['kdj_k']), list(df['kdj_d']), list(df['kdj_j']), \
           list(df['macd']), list(df['macds']), list(df['macdo']), list(df['cci']), list(df['rsi']), \
           list(df['ema5']), list(df['ema10']), list(df['ema20']), \
           list(df['sma5']), list(df['sma10']), list(df['sma20']), \
           list(df['wma5']), list(df['wma10']), list(df['wma20'])

def get_economy_data(economy_file):
    print('get_economy_data')

    df = pd.read_csv(economy_file)

    df['Date'] = df['Date'].astype(str)
    df['Date'] = df['Date'].str.replace(". ", "-", regex=False)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m")
    df = df.set_index('Date')
    return df

def get_date(stock_file):
    print('get_date()')

    df = pd.read_csv(stock_file)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return list(df['date'])

def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
