import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam





class Agent:
    """ Stock Trading Bot """
    # 초기화
    def __init__(self, state_size, pretrained=False, model_name=None):

        # agent config
        self.state_size = state_size*2 + 7*3 	# colse_data 10, volumn_data 10, economy_leading_data 21
        self.action_size = 2
        self.buy_actor_model_name = 'buy_actor_' + model_name
        self.buy_critic_model_name = 'buy_critic_' + model_name
        self.sell_actor_model_name = 'sell_actor_' + model_name
        self.sell_critic_model_name = 'sell_critic_' + model_name
        self.asset = 1e7                    # 현재 보유 현금
        self.origin = 1e7                   # 최초 보유 현금
        self.inventory = []                 # 보유 중인 주식
        self.first_iter = True

        # model config
        self.gamma = 0.95 # affinity for long term reward
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.custom_actor_loss = {"actor_loss": self.actor_loss}
        self.custom_critic_loss = {"critic_loss": self.critic_loss}

        self.advantage = 0
        if pretrained and self.buy_actor_model_name and self.sell_actor_model_name and self.buy_critic_model_name and\
                self.sell_critic_model_name is not None:
            print('load model')
            self.buy_actor_model = self.buy_actor_load()
            self.sell_actor_model = self.sell_actor_load()
            self.buy_critic_model = self.buy_critic_load()
            self.sell_critic_model = self.sell_critic_load()
        else:
            print('create model')
            self.buy_actor_model = self.actor_load()
            self.sell_actor_model = self.actor_load()
            self.buy_critic_model = self.critic_load()
            self.sell_critic_model = self.critic_load()



    def actor_loss(self, y_true, y_pred):
        advantage = self.advantage
        action_prob = K.sum(y_true * y_pred, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        return loss

    def critic_loss(self, y_true, y_pred):
        loss = K.mean(K.square(y_true - y_pred))

        return loss

    # actor model 생성
    def actor_load(self):
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size, activation='softmax'))

        model.compile(loss=self.actor_loss, optimizer=Adam(lr=self.actor_lr))

        return model

    # critic model 생성
    def critic_load(self):
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.critic_loss, optimizer=Adam(lr=self.critic_lr))

        return model

    # actor model action 선택
    def buy_act(self, state):
        action_probs = self.buy_actor_model.predict(state).flatten()
        # np.random.choice(self.action_size, 1, p=action_probs) :
        # action_size 0~1의 범위에서
        # 1개 선택
        # action_probs의 확률에 따라
        return np.random.choice(self.action_size, 1, p=action_probs)[0]

    def sell_act(self, state):
        action_probs = self.sell_actor_model.predict(state).flatten()
        # np.random.choice(self.action_size, 1, p=action_probs) :
        # action_size 0~1의 범위에서
        # 1개 선택
        # action_probs의 확률에 따라
        return np.random.choice(self.action_size, 1, p=action_probs)[0]

    def train_buy_model(self, state, action, reward, next_state, done):
        actor_X, actor_y = [], []
        critic_X, critic_y = [], []

        # 현재 state에서 선택하는 각 행동들의 기대 값
        value = self.buy_critic_model.predict(state)[0]
        # 다음 state에서 선택할 수 있는 행동들의 기대 값
        next_value = self.buy_critic_model.predict(next_state)[0]


        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            # advantage는 실제 값에서 기대 값을 빼준 값
            self.advantage = reward - value
            # target은 이전 dqn, ddqn과 같이 실제 값
            target = [reward]
        else:
            # advantage는 실제 값에서 기대 값을 빼준 값
            self.advantage = (reward + self.gamma * next_value) - value
            # target은 이전 dqn, ddqn과 같이 실제 값
            target = reward + self.gamma * next_value

        actor_X.append(state[0])
        actor_y.append(act[0])

        critic_X.append(state[0])
        critic_y.append(target)

        self.buy_actor_model.compile(loss=self.actor_loss, optimizer=Adam(lr=self.actor_lr))

        actor_loss = self.buy_actor_model.fit(
            np.array(actor_X), np.array(actor_y),
            epochs=1, verbose=0
        ).history["loss"]

        critic_loss = self.buy_critic_model.fit(
            np.array(critic_X), np.array(critic_y),
            epochs=1, verbose=0
        ).history["loss"]

        loss = (actor_loss[0] + critic_loss[0]) / 2

        return loss


    def train_sell_model(self, state, action, reward, next_state, done):
        actor_X, actor_y = [], []
        critic_X, critic_y = [], []

        value = self.sell_critic_model.predict(state)[0]
        next_value = self.sell_critic_model.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            self.advantage = reward - value
            target = [reward]
        else:
            self.advantage = (reward + self.gamma * next_value) - value
            target = reward + self.gamma * next_value

        actor_X.append(state[0])
        actor_y.append(act[0])

        critic_X.append(state[0])
        critic_y.append(target)

        self.sell_actor_model.compile(loss=self.actor_loss, optimizer=Adam(lr=self.actor_lr))

        actor_loss = self.sell_actor_model.fit(
            np.array(actor_X), np.array(actor_y),
            epochs=1, verbose=0
        ).history["loss"]

        critic_loss = self.sell_critic_model.fit(
            np.array(critic_X), np.array(critic_y),
            epochs=1, verbose=0
        ).history["loss"]

        loss = (actor_loss[0] + critic_loss[0]) / 2

        return loss

    def save(self, episode):
        self.buy_actor_model.save("models/{}_{}".format(self.buy_actor_model_name, episode))
        self.sell_actor_model.save("models/{}_{}".format(self.sell_actor_model_name, episode))
        self.buy_critic_model.save("models/{}_{}".format(self.buy_critic_model_name, episode))
        self.sell_critic_model.save("models/{}_{}".format(self.sell_critic_model_name, episode))

    def buy_actor_load(self):
        return load_model("models/" + self.buy_actor_model_name, custom_objects=self.custom_actor_loss)

    def sell_actor_load(self):
        return load_model("models/" + self.sell_actor_model_name, custom_objects=self.custom_actor_loss)

    def buy_critic_load(self):
        return load_model("models/" + self.buy_critic_model_name, custom_objects=self.custom_critic_loss)

    def sell_critic_load(self):
        return load_model("models/" + self.sell_critic_model_name, custom_objects=self.custom_critic_loss)