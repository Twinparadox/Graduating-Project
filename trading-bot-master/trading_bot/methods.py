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


def train_model(agent, episode, data, economy_data, ep_count=100, batch_size=32, window_size=10):
    print('train model')
    total_profit = 0
    data_length = len(data[0]) - 1

    agent.asset = 1e7
    agent.inventory = []
    avg_loss = []
    state = get_state(data[0], data[1], data[2], economy_data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], economy_data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state)

        # BUY
        if action == 1:
            # print("BUY")
            # 가용 가능한 자산이 없으면, 주식을 팔아서 거래 진행
            if agent.asset < data[0][t]:
                # print("SELL AND BUY")
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]
                agent.inventory = []

                bought_sum = np.array(stock_list).sum()
                delta = data[0][t] * nStocks - bought_sum
                agent.asset += data[0][t] * nStocks
                reward = delta / bought_sum * 100
                total_profit += delta

                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.inventory.append([data[0][t], nStocks])

            else:
                # print("BUY")
                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.inventory.append([data[0][t], nStocks])

        # SELL
        elif action == 2:
            if len(agent.inventory) > 0:
                # print("SELL")
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]
                agent.inventory = []

                bought_sum = np.array(stock_list).sum()
                delta = data[0][t] * nStocks - bought_sum
                total_profit += delta
                reward = delta / bought_sum * 100
                agent.asset += data[0][t] * nStocks
            else:
                # print("Cannot SELL")
                reward = 0

        # HOLD
        else:
            # print("HOLD")
            stock_list = []
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]

            bought_sum = np.array(stock_list).sum()
            delta = data[0][t] * nStocks - bought_sum
            if bought_sum > 0:
                reward = delta / bought_sum
            else:
                reward = 0

        # print('reward :', reward, 'delta :', delta, 'asset :', agent.asset)
        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        # 행동을 32번 이상 했을때 학습 시작
        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, economy_data, window_size, debug):
    total_profit = 0
    data_length = len(data[0]) - 1

    history = []
    agent.asset = 1e7
    agent.inventory = []

    state = get_state(data[0], data[1], data[2], economy_data, 0, window_size + 1)

    buy_count, sell_count, hold_count = 0, 0, 0

    for t in range(data_length):
        reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], economy_data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            buy_count += 1
            if agent.asset < data[0][t]:
                # First SELL
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]
                agent.inventory = []

                bought_sum = np.array(stock_list).sum()
                agent.asset += data[0][t] * nStocks
                delta = data[0][t] * nStocks - bought_sum
                reward = delta / bought_sum * 100

                total_profit += delta
                history.append((data[0][t] * nStocks, "SELL"))

                if debug:
                    logging.debug("Sell at: {} {} | Position: {} | Total: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[0][t]), nStocks, format_position(delta), format_position(total_profit), reward,
                    t))

                # BUY
                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.inventory.append([data[0][t], nStocks])
                history.append((data[0][t] * nStocks, "BUY"))

                if debug:
                    logging.debug("Buy, at: {} | Day_Index: {}".format(
                        format_currency(data[0][t]), t))
            else:
                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.inventory.append([data[0][t], nStocks])

                history.append((data[0][t] * nStocks, "BUY"))
                if debug:
                    logging.debug("Buy at: {}, {} | Day_Index: {}".format(format_currency(data[0][t]), nStocks, t))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            sell_count += 1
            if len(agent.inventory) > 0:
                # SELL
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]
                agent.inventory = []

                bought_sum = np.array(stock_list).sum()
                delta = data[0][t] * nStocks - bought_sum
                agent.asset += data[0][t] * nStocks
                reward = delta / bought_sum * 100
                total_profit += delta

                history.append((data[0][t] * nStocks, "SELL"))
                if debug:
                    logging.debug("Sell at: {} {} | Position: {} | Total: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[0][t]), nStocks, format_position(delta), format_position(total_profit), reward,
                    t))
            else:
                # Cannot SELL
                reward = 0
                history.append((data[0][t], "HOLD"))
                if debug:
                    logging.debug("Hold at: {} | Reward: {} | Day_Index: {}".format(
                        format_currency(data[0][t]), reward, t))

        # Pure HOLD
        else:
            hold_count += 1
            stock_list = []
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]

            bought_sum = np.array(stock_list).sum()
            delta = data[0][t] * nStocks - bought_sum

            reward = 0
            history.append((data[0][t], "HOLD"))
            if debug:
                logging.debug("Hold at: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[0][t]), reward, t))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history, buy_count, sell_count, hold_count
