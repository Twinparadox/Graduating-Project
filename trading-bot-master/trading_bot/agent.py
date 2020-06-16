import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
# 전역 변수 문제로 추가한 부분
from tensorflow.python.keras.backend import set_session
sess = tf.Session()
graph = tf.get_default_graph()


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
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # [sit, buy, sell]
        self.model_name = model_name
        self.asset = 1e7  # 현재 보유 현금
        self.origin = 1e7  # 최초 보유 현금
        self.inventory = []  # 보유 중인 주식
        self.memory = deque(maxlen=10000)  # 히스토리
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            print('load model')
            self.model = self.load()
        else:
            print('create model')
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn", "dqn"]:
            print('strategy : ', self.strategy)
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

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

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """

        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            # print('radom action')
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1  # make a definite buy on the first iter

        # 전역 변수 문제로 추가한 부분
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            action_probs = self.model.predict(state)
        # print(action_probs)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        # 학습 데이터 샘플링
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                # 샘플링 된 데이터가 에피소드의 마지막 데이터일 경우 보상을 그대로 저장
                if done:
                    target = reward
                # 샘플링 된 데이터가 에피소드 진행 중의 데이터일 경우
                # 행동에 대한 보상 + 다음 상태에서 얻을 수 있을거라 예측되는 최대보상 * 할인율
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # n_iter 값 변화 없어서 모델 업데이트 되지 않음
        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # n_iter 값 변화 없어서 모델 업데이트 되지 않음
        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][
                        np.argmax(self.model.predict(next_state)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self):
        set_session(sess)
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
