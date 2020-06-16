import os
import logging
import numpy as np
import pandas as pd
import coloredlogs

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)
from trading_bot.ops import (
    get_state
)

from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler

app = Flask(__name__)
api = Api(app)
end = False

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

class Trader(Resource):
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
api.add_resource(Trader, '/trader')

# TODO: 전역변수를 최대한 사용하지 않도록 설계해야 함
### RL Trader ###
agent = []
data = []
data_length = 0

initial_offset = []
window_size = []
model_name = []
args = []
eval_stock = []
debug = []
total_profit = 0
history = []
state = []

done = False
t = 0


### TODO : 기존의 evaluate_model에서 for문만 제거하고 Scheduler가 호출할 시 action 수행
def evaluate_model(agent, data, window_size, state, debug):
    if t < data_length:
        reward = 0
        delta = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state, is_eval=True)

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
    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        profit, _ = evaluate_model(agent, data, window_size, debug)
        show_eval_result(model_name, profit, initial_offset)


### TODO : APSCheduler를 Threading으로 대체할 수 있을지 파악이 필요
count = 0
def sensor():
    global count
    sched.print_jobs()
    print('Count: ' , count)
    count += 1

def action():
    evaluate_model(agent, data, window_size, state, debug)

if __name__ == '__main__':
    #args = docopt(__doc__)
    eval_stock = 'data/SS_2019.csv'
    window_size = 10
    model_name = 'model_debug_50'
    debug = '--debug'

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    ### Initialize Trader
    data = get_stock_data(eval_stock)
    initial_offset = data[1] - data[0]

    t = 0
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent = Agent(window_size, pretrained=True, model_name=model_name)
    agent.asset = 1e7
    agent.inventory = []

    state = get_state(data, 0, window_size + 1)

    ### Start APScheduler.BackgroundScheduler
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(sensor, 'interval', seconds=10)
    sched.start()

    try:
        app.run()

    except KeyboardInterrupt:
        print("Aborted")