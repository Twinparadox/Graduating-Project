import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

import time

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """
    # 초기화
    def __init__(self, state_size, strategy="dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy

        # agent config
        self.state_size = state_size*2  	# normalized previous days, present asset, present price
        self.action_size = 2           		# [sit, buy, sell]
        self.buy_model_name = 'buy_' + model_name
        self.sell_model_name = 'sell_' + model_name
        self.asset = 1e7                    # 현재 보유 현금
        self.origin = 1e7                   # 최초 보유 현금
        self.inventory = []                 # 보유 중인 주식
        self.buy_memory = deque(maxlen=10000)   # 히스토리
        self.sell_memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.buy_model_name = 'buy_' + model_name
        self.sell_model_name = 'sell_' + model_name
        self.gamma = 0.95 # affinity for long term reward
        self.buy_epsilon = 1.0
        self.sell_epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if pretrained and self.buy_model_name and self.sell_model_name is not None:
            print('load model')
            self.buy_model = self.buy_load()
            self.sell_model = self.sell_load()
        else:
            print('create model')
            self.buy_model = self._model()
            self.sell_model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn", "dqn"]:
            print('strategy : ', self.strategy)
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_buy_model = clone_model(self.buy_model)
            self.target_buy_model.set_weights(self.buy_model.get_weights())
            self.target_sell_model = clone_model(self.sell_model)
            self.target_sell_model.set_weights(self.sell_model.get_weights())

    def _model(self):
        print('create model')
        """Creates the model
        """
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def buy_remember(self, state, action, reward, next_state, done):

        self.buy_memory.append((state, action, reward, next_state, done))

    def sell_remember(self, state, action, reward, next_state, done):

        self.sell_memory.append((state, action, reward, next_state, done))


    def buy_act(self, state, is_eval=False):
        # buy action
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.buy_epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        action_probs = self.buy_model.predict(state)
        return np.argmax(action_probs[0])

    def sell_act(self, state, is_eval=False):
        # Sell Action
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.sell_epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        action_probs = self.sell_model.predict(state)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):

        buy_mini_batch = random.sample(self.buy_memory, batch_size)

        buy_X_train, buy_y_train = [], []
        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in buy_mini_batch:
                # 샘플링 된 데이터가 에피소드의 마지막 데이터일 경우 보상을 그대로 저장
                if done:
                    target = reward
                # 샘플링 된 데이터가 에피소드 진행 중의 데이터일 경우
                # 행동에 대한 보상 + 다음 상태에서 얻을 수 있을거라 예측되는 최대보상 * 할인율
                else:
                    # approximate deep q-learning equation
                    if (action == 1):
                        target = reward + self.gamma * np.amax(self.sell_model.predict(next_state)[0])
                    else:
                        target = reward + self.gamma * np.amax(self.buy_model.predict(next_state)[0])
                # estimate q-values based on current state
                q_values = self.buy_model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                buy_X_train.append(state[0])
                buy_y_train.append(q_values[0])

        # update q-function parameters based on huber loss gradient
        buy_loss = self.buy_model.fit(
            np.array(buy_X_train), np.array(buy_y_train),
            epochs=1, verbose=0
        ).history["loss"]

        if self.buy_epsilon > self.epsilon_min:
            self.buy_epsilon *= self.epsilon_decay

        sell_mini_batch = random.sample(self.sell_memory, batch_size)

        sell_X_train, sell_y_train = [], []

        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in sell_mini_batch:
                # 샘플링 된 데이터가 에피소드의 마지막 데이터일 경우 보상을 그대로 저장
                if done:
                    target = reward
                # 샘플링 된 데이터가 에피소드 진행 중의 데이터일 경우
                # 행동에 대한 보상 + 다음 상태에서 얻을 수 있을거라 예측되는 최대보상 * 할인율
                else:
                    if (action == 1):
                        # approximate deep q-learning equation
                        target = reward + self.gamma * np.amax(self.buy_model.predict(next_state)[0])
                    else:
                        target = reward + self.gamma * np.amax(self.sell_model.predict(next_state)[0])
                # estimate q-values based on current state
                q_values = self.sell_model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                sell_X_train.append(state[0])
                sell_y_train.append(q_values[0])

        # TODO : ADD DDQN, Dueling DQN, TDQN
        # update q-function parameters based on huber loss gradient
        sell_loss = self.sell_model.fit(
            np.array(sell_X_train), np.array(sell_y_train),
            epochs=1, verbose=0
        ).history["loss"]

        if self.sell_epsilon > self.epsilon_min:
            self.sell_epsilon *= self.epsilon_decay

        loss = (buy_loss[0] + sell_loss[0])/2

        return loss


    def save(self, episode):
        self.buy_model.save("models/{}_{}".format(self.buy_model_name, episode))
        self.sell_model.save("models/{}_{}".format(self.sell_model_name, episode))

    # buy 모델 불러오기
    def buy_load(self):
        return load_model("models/" + self.buy_model_name, custom_objects=self.custom_objects)

    # sell 모델 불러오기
    def sell_load(self):
        return load_model("models/" + self.sell_model_name, custom_objects=self.custom_objects)