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



def show_train_result(result, val_position, initial_offset, bsh):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}  B,S,H: [{},{},{}]'
                     .format(result[0], result[1], format_position(result[2]), result[3], bsh[0], bsh[1], bsh[2]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})  B,S,H: [{},{},{}]'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3], bsh[0], bsh[1], bsh[2]))


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
    return list(df['close']), list(df['volume']), list(df['date']), \
           list(df['kdj_k']), list(df['kdj_d']), list(df['kdj_j']), \
           list(df['MA5']), list(df['MA10'])

def get_economy_data(economy_file):
    print('get_economy_data')

    df = pd.read_csv(economy_file)

    df['Date'] = df['Date'].astype(str)
    df['Date'] = df['Date'].str.replace(". ", "-", regex=False)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m")
    df = df.set_index('Date')
    return df



def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
