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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def train_model(agent, episode, data, economy_data, ep_count=100, batch_size=32, window_size=10):
    print('train model')
    total_profit = 0
    data_length = len(data[0]) - 1
    agent.reset_every = len(data[0])

    # Reward Debug 용
    buy_count, sell_count, hold_count = 0, 0, 0
    sum_reward = 0
    fig = plt.figure()
    axe = fig.add_subplot(111)
    Y_max = -100
    Y_min = 100
    X, Y = [0], [0]
    X_u, Y_u = [], []
    X_d, Y_d = [], []
    sp, = axe.plot([], [], label='same', ms=2.5, color='k', marker='o', ls='')
    sp_u, = axe.plot([], [], label='up', ms=2.5, color='r', marker='o', ls='')
    sp_d, = axe.plot([], [], label='down', ms=2.5, color='b', marker='o', ls='')
    fig.show()

    agent.asset = 1e7
    agent.ownStocks = 0
    agent.inventory = []
    avg_loss = []
    state = get_state(data[0], data[1], data[2], economy_data, data[3:],\
                      [agent.asset, agent.ownStocks], 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], economy_data, data[3:],\
                               [agent.asset, agent.ownStocks],  t + 1, window_size + 1)

        # select an action
        action = agent.act(state)

        # BUY
        if action == 1:
            # print("BUY")
            # 가용 가능한 자산이 없으면, 주식을 팔아서 거래 진행
            # vs
            # 그냥 홀드해버리기
            '''
            if agent.asset < data[0][t]:
                buy_count += 1
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
                agent.ownStocks = 0
                reward = delta / agent.origin
                total_profit += delta

                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.ownStocks += nStocks
                agent.inventory.append([data[0][t], nStocks])
            '''
            if agent.asset < data[0][t]:
                hold_count += 1
                # print("CANNOT BUY AND HOLD")
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]

                bought_sum = np.array(stock_list).sum()
                delta = data[0][t] * nStocks - bought_sum
                reward = 0

            else:
                buy_count += 1
                # print("BUY")
                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.ownStocks += nStocks
                agent.inventory.append([data[0][t], nStocks])

        # SELL
        elif action == 2:
            if len(agent.inventory) > 0:
                sell_count += 1
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
                reward = delta / agent.origin
                agent.asset += data[0][t] * nStocks
                agent.ownStocks = 0
            else:
                hold_count += 1
                # print("CANNOT SELL AND HOLD")
                reward = 0

        # HOLD
        else:
            hold_count += 1
            # print("HOLD")
            stock_list = []
            nStocks = 0
            for item in agent.inventory:
                stock_list.append(item[0] * item[1])
                nStocks += item[1]

            bought_sum = np.array(stock_list).sum()
            delta = data[0][t] * nStocks - bought_sum
            reward = 0

            #if bought_sum != 0:
                #reward = delta / agent.origin
                #reward /= 100

        # print('reward :', reward, 'delta :', delta, 'asset :', agent.asset)
        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        # 행동을 32번 이상 했을때 학습 시작
        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state
        agent.n_iter += 1

        sum_reward += reward
        Y_min = min(Y_min, sum_reward)
        Y_max = max(Y_max, sum_reward)

        if reward > 0:
            X_u.append(t)
            Y_u.append(sum_reward)
        elif reward < 0:
            X_d.append(t)
            Y_d.append(sum_reward)
        else:
            X.append(t)
            Y.append(sum_reward)

        sp.set_data(X,Y)
        sp_d.set_data(X_d, Y_d)
        sp_u.set_data(X_u, Y_u)
        axe.set_xlim(0,data_length)
        axe.set_ylim(Y_min,Y_max)
        axe.set_title("R = {0:.2f}, R_Sum = {1:.2f}, Buy = {2}, Sell = {3}, Hold = {4}".
                      format(reward, sum_reward, buy_count, sell_count, hold_count))
        fig.canvas.draw()
        fig.canvas.flush_events()

        # sum_reward가 -0.2를 넘어가면, 순간적으로 손해가 큼
        # 해당 에피소드는 종료하고, 새 에피소드 진행
        #if sum_reward < -0.2:
        #    break

    if episode % 10 == 0:
        agent.save(episode)
    plt.close()

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, economy_data, window_size, debug):
    total_profit = 0
    data_length = len(data[0]) - 1

    history = []
    agent.asset = 1e7
    agent.inventory = []
    agent.ownStocks = 0

    state = get_state(data[0], data[1], data[2], economy_data, data[3:],\
                      [agent.asset, agent.ownStocks], 0, window_size + 1)

    buy_count, sell_count, hold_count = 0, 0, 0

    for t in range(data_length):
        reward = 0
        delta = 0
        next_state = get_state(data[0], data[1], data[2], economy_data, data[3:],\
                               [agent.asset, agent.ownStocks], t + 1, window_size + 1)

        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            # print("BUY")
            # 가용 가능한 자산이 없으면, 주식을 팔아서 거래 진행
            # vs
            # 그냥 홀드해버리기
            '''
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
                agent.ownStocks = 0
                delta = data[0][t] * nStocks - bought_sum
                reward = delta / agent.origin

                total_profit += delta
                history.append((data[0][t] * nStocks, "SELL"))

                if debug:
                    logging.debug("Sell at: {} {} | Position: {} | Total: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[0][t]), nStocks, format_position(delta), format_position(total_profit), reward,
                    t))

                # BUY
                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.ownStocks += nStocks
                agent.inventory.append([data[0][t], nStocks])
                history.append((data[0][t] * nStocks, "BUY"))

                if debug:
                    logging.debug("Buy, at: {} | Day_Index: {}".format(
                        format_currency(data[0][t]), t))
            '''
            if agent.asset < data[0][t]:
                hold_count += 1
                # print("CANNOT BUY AND HOLD")
                stock_list = []
                nStocks = 0
                for item in agent.inventory:
                    stock_list.append(item[0] * item[1])
                    nStocks += item[1]

                bought_sum = np.array(stock_list).sum()
                delta = data[0][t] * nStocks - bought_sum
                reward = 0

                if debug:
                    logging.debug("Cannot Buy & Hold at: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[0][t]), reward, t))

            else:
                buy_count += 1
                nStocks = agent.asset // data[0][t]
                agent.asset -= nStocks * data[0][t]
                agent.ownStocks += nStocks
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
                agent.ownStocks = 0
                reward = delta / agent.origin
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

            # if bought_sum != 0:
            #     reward = delta / agent.origin
            #     reward /= 100

            history.append((data[0][t], "HOLD"))
            if debug:
                logging.debug("Hold at: {} | Reward: {} | Day_Index: {}".format(
                    format_currency(data[0][t]), reward, t))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        agent.n_iter += 1

        state = next_state
        if done:
            return total_profit, history, buy_count, sell_count, hold_count
