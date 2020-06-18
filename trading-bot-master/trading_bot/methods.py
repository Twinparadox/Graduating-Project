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
    state = get_state(data, 0, window_size + 1)

    buy_hold = 0
    buy_act = 0
    sell_hold = 0
    sell_act = 0

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        sell_reward = 0
        buy_reward = 0
        delta = 0
        next_state = get_state(data, t+1, window_size + 1)

        # select an action
        if (data[t] <= agent.asset):
            buy_t = t
            buy_state = state
            buy_action = agent.buy_act(buy_state)
            buy_next_state = get_state(data, t + 1, window_size + 1)
            buy_done = (t == data_length - 1)

            if buy_action == 1:
                buy_act += 1
                nStocks = agent.asset // data[t]

                agent.asset -= nStocks * data[t]
                agent.inventory.append([data[t], nStocks])
            else:
                buy_hold += 1
                agent.buy_remember(buy_state, buy_action, buy_reward, buy_next_state, buy_done)
                pass
        else:
            sell_state = state
            sell_action = agent.sell_act(sell_state)
            sell_next_state = get_state(data, t + 1, window_size + 1)
            sell_done = (t == data_length - 1)

            if sell_action == 1:
                sell_act += 1
                stock_list = []

                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]
                agent.inventory = []

                bought_sum = np.array(stock_list).sum()

                delta = data[t] * nStocks - bought_sum

                agent.asset += data[t] * nStocks

                sell_reward = delta / bought_sum

                buy_reward = sell_reward
                total_profit += delta

                agent.buy_remember(buy_state, buy_action, buy_reward, buy_next_state, buy_done)
                agent.sell_remember(sell_state, sell_action, sell_reward, sell_next_state, sell_done)
            else:
                sell_hold += 1
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]

                bought_sum = np.array(stock_list).sum()
                delta = data[t] * nStocks - bought_sum
                sell_reward = delta / bought_sum

                agent.sell_remember(sell_state, sell_action, sell_reward, sell_next_state, sell_done)




        # 행동을 32번 이상 했을때 학습 시작
        if len(agent.buy_memory) > batch_size and len(agent.sell_memory) > batch_size:
            #
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state
        done = (t == data_length - 1)

        if (done):
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]
            agent.inventory = []

            agent.asset += data[t] * nStocks

    agent.save(episode)
    print("Buy act : ", buy_act, " Buy Hold : ", buy_hold, " sell Act : ", sell_act, " Sell Hold : ", sell_hold)
    return (episode, ep_count, total_profit, agent.asset, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.asset = 1e7
    agent.inventory = []
    
    state = get_state(data, 0, window_size + 1)

    buy_hold = 0
    buy_act = 0
    sell_hold = 0
    sell_act = 0

    for t in range(data_length):        
        reward = 0
        delta = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # BUY
        if agent.asset >= data[t]:
            action = agent.buy_act(state, is_eval=True)

            if action == 1:
                buy_act += 1
                nStocks = agent.asset // data[t]

                agent.asset -= nStocks * data[t]
                agent.inventory.append([data[t], nStocks])

                history.append((data[t] * nStocks, "BUY"))
                if debug:
                    logging.debug("Buy at: {}, {} | Day_Index: {}".format(format_currency(data[t]), nStocks, t))
            else:
                buy_hold += 1
                history.append((data[t], "Buy HOLD"))
                if debug:
                    logging.debug("Buy Hold at: {} | Day_Index: {}".format(
                        format_currency(data[t]), t)
                    )
        # SELL
        else:
            action = agent.sell_act(state, is_eval=True)

            if action == 1:
                sell_act += 1
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]
                agent.inventory = []

                bought_sum = np.array(stock_list).sum()
                delta = data[t] * nStocks - bought_sum
                agent.asset += data[t] * nStocks
                reward = delta / bought_sum
                total_profit += delta
                history.append((data[t] * nStocks, "SELL"))

                if debug:
                    logging.debug("Sell at: {} {} | Position: {} | Total {} | Reward: {} | Day_Index: {}".format(
                        format_currency(data[t]), nStocks, format_position(delta), format_position(total_profit),
                        reward, t
                    ))
            else:
                sell_hold += 1
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]

                bought_sum = np.array(stock_list).sum()
                delta = data[t] * nStocks - bought_sum

                reward = delta / bought_sum

                history.append((data[t], "Sell HOLD"))
                if debug:
                    logging.debug("Hold at: {} | Reward: {} | Day_Index: {}".format(
                        format_currency(data[t]), reward, t))

        done = (t == data_length - 1)
        # agent.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            if debug:
                logging.debug("Buy Hold : {} | Buy Act : {} | Sell Hold : {} | Sell Act : {}".format(
                    buy_hold, buy_act, sell_hold, sell_act))

            return total_profit, history

