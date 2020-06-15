"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]

Options:
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
"""

import os
import logging
import numpy as np
import pandas as pd
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)
from .ops import (
    get_state
)

from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse

app = Flask(__name__)
api = Api(app)

class CreateUser(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('email', type=str)
            parser.add_argument('user_name', type=str)
            parser.add_argument('password', type=str)
            args = parser.parse_args()

            _userEmail = args['email']
            _userName = args['user_name']
            _userPassword = args['password']
            return {'Email': args['email'], 'UserName': args['user_name'], 'Password': args['password']}
        except Exception as e:
            return {'error': str(e)}

api.add_resource(CreateUser, '/user')


### RL Trader ###
def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.asset = 1e7
    agent.inventory = []

    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):
        reward = 0
        delta = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state, is_eval=True)

        # # BUY
        # if action == 1:
        #     agent.inventory.append(data[t])
        #
        #     history.append((data[t], "BUY"))
        #     if debug:
        #         logging.debug("Buy at: {} | Day_Index: {}".format(format_currency(data[t]), t))
        #
        # # SELL
        # elif action == 2 and len(agent.inventory) > 0:
        #     stock_list = []
        #     for i in agent.inventory:
        #         stock_list.append(i)
        #     agent.inventory = []
        #
        #     bought_sum = np.array(stock_list).sum()
        #
        #     delta = 0
        #     for bought_price in stock_list:
        #         delta += data[t] - bought_price
        #
        #     reward = float(delta) / float(bought_sum) * 100
        #
        #     total_profit += delta
        #
        #     history.append((data[t], "SELL"))
        #     if debug:
        #         logging.debug("Sell at: {} | Position: {} | Total: {} | Reward: {} | Day_Index: {}".format(
        #             format_currency(data[t]), format_position(delta), format_position(total_profit), format_position(reward), t))

        # BUY
        if action == 1:
            if agent.asset < data[t]:
                history.append((data[t], "HOLD"))
                if debug:
                    logging.debug("Cannot Buy, Hold at: {} | Day_Index: {}".format(
                        format_currency(data[t]), t))

            else:
                nStocks = agent.asset // data[t]

                if nStocks == 0:
                    nStocks = agent.asset // data[t]

                agent.asset -= nStocks * data[t]
                agent.inventory.append([data[t], nStocks])

                history.append((data[t] * nStocks, "BUY"))
                if debug:
                    logging.debug("Buy at: {}, {} | Day_Index: {}".format(format_currency(data[t]), nStocks, t))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            stock_list = []
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]
            agent.inventory = []

            bought_sum = np.array(stock_list).sum()

            delta = data[t] * nStocks - bought_sum

            agent.asset += data[t] * nStocks

            reward = delta / bought_sum * 100

            total_profit += delta

            history.append((data[t] * nStocks, "SELL"))
            if debug:
                logging.debug("Sell at: {} {} | Position: {} | Total: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[t]), nStocks, format_position(delta), format_position(total_profit), reward,
                    t))

        # HOLD
        else:
            stock_list = []
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]

            bought_sum = np.array(stock_list).sum()
            delta = data[t] * nStocks - bought_sum

            if bought_sum > 0:
                reward = delta / bought_sum
            else:
                reward = 0
            history.append((data[t], "HOLD"))
            if debug:
                logging.debug("Hold at: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[t]), reward, t))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history

def main(eval_stock, window_size, model_name, debug):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """
    data = get_stock_data(eval_stock)
    initial_offset = data[1] - data[0]

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        profit, _ = evaluate_model(agent, data, window_size, debug)
        show_eval_result(model_name, profit, initial_offset)


if __name__ == '__main__':
    #args = docopt(__doc__)
    #eval_stock = args["<eval-stock>"]
    #window_size = int(args["--window-size"])
    #model_name = args["--model-name"]
    #debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        app.run(debug=True)

    except KeyboardInterrupt:
        print("Aborted")