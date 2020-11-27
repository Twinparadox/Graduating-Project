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
import matplotlib.pyplot as plt
import csv

def train_model(agent, episode, data, economy_data, ep_count=100, batch_size=32, window_size=10, last_checkpoint=0, model_name=''):
    print('train model')
    total_profit = 0
    data_length = len(data[0]) - 1

    agent.asset = 1e7
    agent.inventory = []
    avg_loss = []
    state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                      data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], 0, window_size + 1)

    buy_hold = 0
    buy_act = 0
    sell_hold = 0
    sell_act = 0

    asset = [agent.asset]
    close = []

    f = open(model_name + 'train_result_' + str(episode) + '.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['asset', 'close'])
    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        sell_reward = 0
        buy_reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                               data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], t+1, window_size + 1)
        close.append(data[1][t])
        # select an action
        if (data[1][t] <= agent.asset):

            buy_state = state
            buy_action = agent.buy_act(buy_state)
            buy_next_state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                                       data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], t + 1, window_size + 1)
            buy_done = (t == data_length - 1)

            if buy_action == 1:

                buy_act += 1
                nStocks = agent.asset // data[1][t]

                agent.asset -= nStocks * data[1][t]
                agent.inventory.append([data[1][t], nStocks])

                buy_delta = data[1][t+1] - data[1][t]
                buy_reward = buy_delta

                agent.buy_remember(buy_state, buy_action, buy_reward, buy_next_state, buy_done)
            else:
                buy_hold += 1

                hold_delta = data[1][t] - data[1][t+1]
                hold_reward = hold_delta/data[1][t]
                if(hold_reward > 0):
                    agent.buy_remember(buy_state, buy_action, hold_reward, buy_next_state, buy_done)
                pass
        else:
            sell_state = state
            sell_action = agent.sell_act(sell_state)
            sell_next_state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                                        data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], t + 1, window_size + 1)
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

                delta = data[1][t] * nStocks - bought_sum

                agent.asset += data[1][t] * nStocks

                sell_reward = delta/bought_sum

                total_profit += delta
                if(sell_reward > 0):
                    agent.sell_remember(sell_state, sell_action, sell_reward, sell_next_state, sell_done)
            else:
                sell_hold += 1
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]

                bought_sum = np.array(stock_list).sum()

                next_delta = data[1][t + 1] * nStocks - bought_sum

                hold_reward = next_delta/bought_sum
                if(hold_reward > 0):
                    agent.sell_remember(sell_state, sell_action, hold_reward, sell_next_state, sell_done)

        state = next_state
        done = (t == data_length - 1)

        # 행동을 32번 이상 했을때 학습 시작
        if len(agent.buy_memory) > batch_size and len(agent.sell_memory) > batch_size:
            if done:
                loss = agent.train_experience_replay(batch_size, True)
            else:
                loss = agent.train_experience_replay(batch_size, False)
            avg_loss.append(loss)

        if (done):
            nStocks = 0
            for item in agent.inventory:
                nStocks += item[1]
            agent.inventory = []

            agent.asset += data[1][t] * nStocks

        num_Stocks = 0
        for item in agent.inventory:
            num_Stocks += item[1]


        asset.append(agent.asset + data[1][t] * num_Stocks)
        wr.writerow([agent.asset + data[1][t] * num_Stocks, data[1][t]])

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(asset)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(close)
    plt.savefig(model_name + ' train asset and close' + str(episode))
    plt.close(fig)



    agent.save(episode)
    print("Buy act : ", buy_act, " Buy Hold : ", buy_hold, " sell Act : ", sell_act, " Sell Hold : ", sell_hold)
    return (episode, ep_count, total_profit, agent.asset, np.mean(np.array(avg_loss))), False


def evaluate_model(agent, data, economy_data, window_size, debug, episode, model_name=''):
    total_profit = 0
    data_length = len(data[0]) - 1

    history = []
    agent.asset = 1e7
    agent.inventory = []
    
    state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                      data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], 0, window_size + 1)
    buy_hold = 0
    buy_act = 0
    sell_hold = 0
    sell_act = 0

    asset = []
    close = []

    f = open(model_name + 'test_result_' + str(episode) + '.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['asset', 'close'])

    for t in range(data_length):        
        reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                               data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], t + 1, window_size + 1)
        close.append(data[1][t])
        # BUY
        if agent.asset >= data[1][t]:
            action = agent.buy_act(state, is_eval=True)

            if action == 1:
                buy_act += 1
                nStocks = agent.asset // data[1][t]

                agent.asset -= nStocks * data[1][t]
                agent.inventory.append([data[1][t], nStocks])

                history.append((data[1][t] * nStocks, "BUY"))
                if debug:
                    logging.debug("Buy at: {}, {} | Day_Index: {}".format(format_currency(data[1][t]), nStocks, t))
            else:
                buy_hold += 1
                history.append((data[1][t], "Buy HOLD"))
                if debug:
                    logging.debug("Buy Hold at: {} | Day_Index: {}".format(
                        format_currency(data[1][t]), t)
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
                delta = data[1][t] * nStocks - bought_sum
                agent.asset += data[1][t] * nStocks
                reward = delta / bought_sum
                total_profit += delta
                history.append((data[1][t] * nStocks, "SELL"))

                if debug:
                    logging.debug("Sell at: {} {} | Position: {} | Total {} | Reward: {} | Day_Index: {}".format(
                        format_currency(data[1][t]), nStocks, format_position(delta), format_position(total_profit),
                        reward, t
                    ))
            else:
                sell_hold += 1

                history.append((data[1][t], "Sell HOLD"))
                if debug:
                    logging.debug("Hold at: {} | Day_Index: {}".format(
                        format_currency(data[1][t]), t))

        done = (t == data_length - 1)

        state = next_state

        num_Stocks = 0
        for item in agent.inventory:
            num_Stocks += item[1]
        asset.append(agent.asset + data[1][t] * num_Stocks)

        wr.writerow([agent.asset + data[1][t] * num_Stocks, data[1][t]])

        if done:
            stock_list = []
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]
            agent.inventory = []

            bought_sum = np.array(stock_list).sum()
            delta = data[1][t] * nStocks - bought_sum
            total_profit += delta
            agent.asset += data[1][t] * nStocks

            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(asset)
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(close)
            plt.savefig(model_name + 'test_result_' + str(episode))
            plt.close(fig)

            f.close()

            if debug:
                logging.debug("Buy Hold : {} | Buy Act : {} | Sell Hold : {} | Sell Act : {}".format(
                    buy_hold, buy_act, sell_hold, sell_act))

            return total_profit, agent.asset, history

