import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


import time

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    print('train model')
    total_profit = 0
    data_length = len(data) - 1

    agent.asset = 1e7
    agent.inventory = []
    avg_loss = []
    state = get_state(data, 0, window_size + 1, agent.asset)


    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):        
        reward = 0
        delta = 0
        next_state = get_state(data, t + 1, window_size + 1, agent.asset)

        # select an action
        if (data[t] < agent.asset):
            action = agent.buy_act(state)

            if action == 1:
                nStocks = agent.asset // data[t]

                agent.asset -= nStocks * data[t]
                agent.inventroy.append([data[t], nStocks])

            else:
                stock_list = []
                nStocks = 0

                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]


        else:
            action = agent.sell_act(state)

        # if action == 1:
        #     agent.inventory.append(data[t])
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

        # # BUY
        # if action == 1 and agent.asset < data[t]:
        #     # print("BUY")
        #
        #     nStocks = agent.asset // data[t]
        #
        #     agent.asset -= nStocks * data[t]
        #     agent.inventory.append([data[t], nStocks])
        #     '''
        #     if agent.asset < data[t]:
        #         pass
        #     else:
        #         nStocks = agent.asset // data[t]
        #
        #         if nStocks == 0:
        #             nStocks = agent.asset // data[t]
        #
        #         agent.asset -= nStocks * data[t]
        #         agent.inventory.append([data[t], nStocks])
        #     '''
        # # SELL
        # elif action == 2 and len(agent.inventory) > 0:
        #     # print("SELL")
        #     stock_list = []
        #     nStocks = 0
        #     for item in agent.inventory:
        #         stock_list.append(item[0] * item[1])
        #         nStocks += item[1]
        #     agent.inventory = []
        #
        #     bought_sum = np.array(stock_list).sum()
        #
        #     delta = data[t] * nStocks - bought_sum
        #
        #     agent.asset += data[t] * nStocks
        #
        #     reward = delta / bought_sum * 100
        #
        #     total_profit += delta
        #
        # # HOLD
        # else:
        #     # print("HOLD")
        #     stock_list = []
        #     nStocks = 0
        #     for item in agent.inventory:
        #         stock_list.append(item[0] * item[1])
        #         nStocks += item[1]
        #
        #     bought_sum = np.array(stock_list).sum()
        #     delta = data[t] * nStocks - bought_sum
        #     if bought_sum > 0:
        #         reward = delta / bought_sum
        #     else:
        #         reward = 0

        # print('reward :', reward, 'delta :', delta, 'asset :', agent.asset)
        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        # 행동을 32번 이상 했을때 학습 시작
        if len(agent.memory) > batch_size:
            #
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.asset = 1e7
    agent.inventory = []
    
    state = get_state(data, 0, window_size + 1, agent.asset)

    for t in range(data_length):        
        reward = 0
        delta = 0
        next_state = get_state(data, t + 1, window_size + 1, agent.asset)
        
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
                    format_currency(data[t]), nStocks, format_position(delta), format_position(total_profit), reward, t))

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
