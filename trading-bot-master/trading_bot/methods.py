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

from keras.models import load_model, clone_model

import time

def train_model(agent, episode, data, economy_data, ep_count=100, batch_size=32, window_size=10, last_checkpoint=0):
    print('train model')
    total_profit = 0
    data_length = len(data[0]) - 1

    agent.asset = 1e7
    agent.inventory = []
    state = get_state(data[0], data[1], data[2], economy_data, 0, window_size + 1)

    buy_hold = 0
    buy_act = 0
    sell_hold = 0
    sell_act = 0

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        sell_reward = 0
        buy_reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], economy_data, t+1, window_size + 1)

        # select an action
        if (data[0][t] <= agent.asset):
            buy_t = t
            buy_state = state
            buy_action = agent.buy_act(buy_state)

            buy_next_state = get_state(data[0], data[1], data[2], economy_data, t + 1, window_size + 1)
            buy_done = (t == data_length - 1)

            if buy_action == 1:
                buy_act += 1
                nStocks = agent.asset // data[0][t]

                agent.asset -= nStocks * data[0][t]
                agent.inventory.append([data[0][t], nStocks])
            else:
                buy_hold += 1
                agent.train_buy_model(buy_state, buy_action, buy_reward, buy_next_state, buy_done)
                pass
        else:
            sell_state = state
            sell_action = agent.sell_act(sell_state)

            sell_next_state = get_state(data[0], data[1], data[2], economy_data, t + 1, window_size + 1)
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

                delta = data[0][t] * nStocks - bought_sum

                agent.asset += data[0][t] * nStocks

                sell_reward = delta / bought_sum

                buy_reward = sell_reward
                total_profit += delta

                agent.train_buy_model(buy_state, buy_action, buy_reward, buy_next_state, buy_done)
                agent.train_sell_model(sell_state, sell_action, sell_reward, sell_next_state, sell_done)
            else:
                sell_hold += 1
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]

                bought_sum = np.array(stock_list).sum()
                delta = data[0][t] * nStocks - bought_sum

                agent.train_sell_model(sell_state, sell_action, sell_reward, sell_next_state, sell_done)

        # 현재 이익(total_profit)이 원금의 10% 이상 손실본 경우
        if total_profit <= -0.1 * agent.origin:
            nStocks = 0
            for item in agent.inventory:
                nStocks += item[1]
            currency_asset = agent.asset + nStocks*data[0][t]
            agent.asset = currency_asset

            if last_checkpoint == 0: # 저장된 에피소드가 없을 때
                agent.buy_actor_model = agent.actor_load()
                agent.sell_actor_model = agent.actor_load()
                agent.buy_critic_model = agent.critic_load()
                agent.sell_critic_model = agent.critic_load()
            else: # 가장 마지막에 저장된 에피소드
                agent.buy_actor_model = load_model("models/{}_{}".format(agent.buy_actor_model_name, last_checkpoint))
                agent.sell_actor_model = load_model("models/{}_{}".format(agent.sell_actor_model_name, last_checkpoint))
                agent.buy_critic_model = load_model("models/{}_{}".format(agent.buy_critic_model_name, last_checkpoint))
                agent.sell_critic_model = load_model("models/{}_{}".format(agent.sell_critic_mode_name, last_checkpoint))

            print("Early Stoping - {} episode, last_checkpoint {}".format(episode, last_checkpoint))
            print(episode, ep_count, total_profit, agent.asset)
            return (episode, ep_count, total_profit, agent.asset), True


        state = next_state
        done = (t == data_length - 1)

        if (done):
            nStocks = 0
            for item in agent.inventory:
                nStocks += item[1]
            agent.inventory = []

            agent.asset += data[0][t] * nStocks

    agent.save(episode)
    print("Buy act : ", buy_act, " Buy Hold : ", buy_hold, " sell Act : ", sell_act, " Sell Hold : ", sell_hold)
    return (episode, ep_count, total_profit, agent.asset), False


def evaluate_model(agent, data, economy_data, window_size, debug):
    total_profit = 0
    data_length = len(data[0]) - 1

    history = []
    agent.asset = 1e7
    agent.inventory = []
    
    state = get_state(data[0], data[1], data[2], economy_data, 0, window_size + 1)

    buy_hold = 0
    buy_act = 0
    sell_hold = 0
    sell_act = 0

    for t in range(data_length):        
        reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], economy_data, t + 1, window_size + 1)

        # BUY
        if agent.asset >= data[0][t]:
            action = agent.buy_act(state)

            if action == 1:
                buy_act += 1
                nStocks = agent.asset // data[0][t]

                agent.asset -= nStocks * data[0][t]
                agent.inventory.append([data[0][t], nStocks])

                history.append((data[0][t] * nStocks, "BUY"))
                if debug:
                    logging.debug("Buy at: {}, {} | Day_Index: {}".format(format_currency(data[0][t]), nStocks, t))
            else:
                buy_hold += 1
                history.append((data[0][t], "Buy HOLD"))
                if debug:
                    logging.debug("Buy Hold at: {} | Day_Index: {}".format(
                        format_currency(data[0][t]), t)
                    )
        # SELL
        else:
            action = agent.sell_act(state)

            if action == 1:
                sell_act += 1
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]
                agent.inventory = []

                bought_sum = np.array(stock_list).sum()
                delta = data[0][t] * nStocks - bought_sum
                agent.asset += data[0][t] * nStocks
                reward = delta / bought_sum
                total_profit += delta
                history.append((data[0][t] * nStocks, "SELL"))

                if debug:
                    logging.debug("Sell at: {} {} | Position: {} | Total {} | Reward: {} | Day_Index: {}".format(
                        format_currency(data[0][t]), nStocks, format_position(delta), format_position(total_profit),
                        reward, t
                    ))
            else:
                sell_hold += 1

                history.append((data[0][t], "Sell HOLD"))
                if debug:
                    logging.debug("Hold at: {} | Day_Index: {}".format(
                        format_currency(data[0][t]), t))

        done = (t == data_length - 1)

        state = next_state

        if done:
            if debug:
                logging.debug("Buy Hold : {} | Buy Act : {} | Sell Hold : {} | Sell Act : {}".format(
                    buy_hold, buy_act, sell_hold, sell_act))

            return total_profit, history

