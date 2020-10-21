import os
import logging
import numpy as np
import pandas as pd
import coloredlogs

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    get_economy_data,
    get_date,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)
from trading_bot.ops import (
    get_state
)
from db_utils.utils import (
    connect_server,
    disconnect_server,
    insert_data
)

from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler

app = Flask(__name__)
CORS(app)
api = Api(app)
end = False

parser = reqparse.RequestParser()


# TODO 거래 기록을 DB에서 가져오게 하는 것이 타당할 듯, 현재는 History에서 보내는 중
class Trader(Resource):
    def get(self):
        try:
            # parser.add_argument('email', type=str)
            # parser.add_argument('user_name', type=str)
            # parser.add_argument('password', type=str)
            # args = parser.parse_args()
            #
            # _userEmail = args['email']
            # _userName = args['user_name']
            # _userPassword = args['password']
            return {'History': history[-10:], 'Buy': num_buy, 'Sell':num_sell, 'CannotBuy':num_cannotbuy,
                     'CannotSell':num_cannotsell, 'Hold':num_hold}
        except Exception as e:
            return {'error': str(e)}

    def post(self):
        try:
            parser.add_argument('names', type=str)
            parser.add_argument('days', type=int)
            args = parser.parse_args()

            _corpName = args['names']
            _days = args['days']
            return {'History': history[-_days:], 'Buy': num_buy, 'Sell':num_sell, 'CannotBuy':num_cannotbuy,
                     'CannotSell':num_cannotsell, 'Hold':num_hold}
        except Exception as e:
            return {'error': str(e)}


#TODO 주식, 경제 지표 조회용 API, DB와 연동 필요할 듯
class Stock(Resource):
    def get(self):
        pass

    def post(self):
        pass


api.add_resource(Trader, '/trader')
api.add_resource(Stock, '/stock')

# TODO: 전역변수를 최대한 사용하지 않도록 설계해야 함
### RL Trader ###
#initial_offset = []
window_size = []
model_name = []
args = []
eval_stock = []
debug = []
total_profit = 0
history = []

t = 0

num_buy = 0
num_sell = 0
num_buy_hold = 0
num_sell_hold = 0
num_hold = 0

### TODO : 시각화 진행 및 로그 출력
### TODO : 서버 활용할 경우 서버 접속 필요
def trade_stock(agent, data, date_list, window_size, state, debug):
    global total_profit
    global num_buy, num_sell, num_buy_hold, num_sell_hold

    profit = 0
    reward = 0
    delta = 0
    next_state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], economy_data, t+1, window_size + 1)
    print(state)
    if agent.asset >= data[3][t]:
        action = agent.buy_act(state)

        if action == 1:
            num_buy += 1
            nStocks = agent.asset // data[3][t]

            agent.asset -= nStocks * data[3][t]
            agent.inventory.append([data[3][t], nStocks])

            history.append({"index":t, "date":date_list[t], "price":data[3][t], "nStocks":nStocks, "volume":nStocks, "action":"Buy"})
            if debug:
                logging.debug("Buy at: {}, {} | Day_Index: {}".format(format_currency(data[3][t]), nStocks, t))
        else:
            num_buy_hold += 1
            history.append({"index":t, "date":date_list[t], "price":data[3][t], "volume":0, "action":"Buy Hold"})
            if debug:
                logging.debug("Buy hold, Hold at: {} | Day_Index: {}".format(
                    format_currency(data[3][t]), t))

    else:
        action = agent.sell_act(state)

        if action == 1:
            num_sell += 1
            stock_list = []
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]
            agent.inventory = []

            bought_sum = np.array(stock_list).sum()
            delta = data[3][t] * nStocks - bought_sum
            agent.asset += data[3][t] * nStocks
            reward = delta / bought_sum
            total_profit += delta

            history.append({"index": t, "date": date_list[t], "price": data[3][t], "nStocks": nStocks, "volume": nStocks,
                            "action": "Sell"})
            if debug:
                logging.debug("Sell at: {} {} | Position: {} | Total: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[3][t]), nStocks, format_position(delta), format_position(total_profit), reward,
                    t))
        else:
            num_sell_hold += 1
            history.append({"index": t, "date": date_list[t], "price": data[3][t], "volume": 0, "action": "Sell Hold"})
            if debug:
                logging.debug("Sell Hold, Hold at: {} | Day_Index: {}".format(format_currency(data[3][t]), t))



    # done = (t == data_length - 1)
    # agent.memory.append((state, action, reward, next_state, done))

    state = next_state

    insert_data("SS", history[-1], delta, total_profit)

    return profit, next_state

### TODO : APSCheduler를 Threading으로 대체할 수 있을지 파악이 필요
count = 0
def time_lapse(**kwargs):
    global count
    global t
    global state

    sched.print_jobs()
    print('Count: ', count)
    count += 1

    agent = kwargs['agent']
    data = kwargs['data']
    date_list = kwargs['date_list']
    window_size = kwargs['window_size']
    debug = kwargs['debug']

    # 거래 진행하면, DB에 넣도록 구현
    if t < data_length - 1:
        profit, next_state = trade_stock(agent, data, date_list, window_size, state, debug)
        state = next_state

        t += 1
    
    else:
        print("Cannot Trade Anymore")

if __name__ == '__main__':
    #args = docopt(__doc__)
    eval_stock = 'data/SS_2019.csv'
    economy = 'data/economy_leading_2005.csv'
    model_name = 'model_debug_2'
    debug = '--debug'

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    ### TODO: 초기화 해주는 함수 있으면 좋을 듯
    ### Initialize Trader
    window_size = 5
    data = get_stock_data(eval_stock)
    date_list = get_date(eval_stock)
    economy_data = get_economy_data(economy)
    initial_offset = data[3][1] - data[3][0]
    data_length = len(data[0]) - 1
    print("data_length :", data_length)
    agent = Agent(window_size, pretrained=True, model_name=model_name)
    agent.asset = 1e7
    agent.inventory = []

    state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], economy_data, 0, window_size + 1)

    ### TODO: args가 과연 필요한가에 대한 파악 필요
    kwargs = {"agent":agent, "data":data, "date_list":date_list, "window_size":window_size, "debug":debug}

    connect_server()

    ### Start APScheduler.BackgroundScheduler
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(time_lapse, 'interval', kwargs=kwargs, seconds=3)
    sched.start()

    try:
        app.run()

    except KeyboardInterrupt:
        print("Aborted")
        disconnect_server()