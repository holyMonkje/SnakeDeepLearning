from ple.games.snake import Snake
from ple import PLE
import numpy as np
import time
from collections import deque
import random
import pandas as pd
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def dict_to_list(dictObj):
    return np.array([int(dictObj['snake_head_x']), int(dictObj['snake_head_x']), int(dictObj['food_x']), int(dictObj['food_y'])])


class NaiveAgent():

    def __init__(self, actions):

        self.epsilon = 1
        self.learning_rate = 0.001
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.actions = actions
        self.action_size = len(actions)
        self.gamma = 0.95    # discount rate
        self.model = self._build_model()
        self.memory = deque(maxlen=2000)

        print(actions)

    def _build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def pickAction(self, reward, obs, state):

        state = dict_to_list(state)

        if np.random.rand() <= self.epsilon:
            return self.actions[np.random.randint(0, len(self.actions))]

        act_values = self.model.predict(np.array([state]))

        print(act_values)

        return self.actions[np.argmax(act_values[0])]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((dict_to_list(state), action,
                            reward, dict_to_list(next_state), done))

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))

            target_f[0][self.actions.index(action)] = target

            self.model.fit(np.array([state]), np.array(
                target_f), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


game = Snake(width=320, height=320)
p = PLE(game, fps=30, display_screen=True)
agent = NaiveAgent(p.getActionSet())

p.init()
reward = 0.0

learning_time = 100000

actions = []
rewards = []
snake_head_x = []
snake_head_y = []
food_x = []
food_y = []

learning_sample = pd.DataFrame()

observation = p.getScreenRGB()
state = p.game.getGameState()

del state['snake_body']
del state['snake_body_pos']

for i in range(learning_time):

    if p.game_over():
        p.reset_game()

    action = agent.pickAction(reward, observation, state)
    reward = p.act(action)

    observation = p.getScreenRGB()
    next_state = p.game.getGameState()

    done = (reward == -5)

    agent.remember(state, action, reward, next_state, done)

    state = next_state

    del state['snake_body']
    del state['snake_body_pos']

    if done:
        agent.replay(100)

    # {'snake_head_x': 339.20000000000005, 'snake_head_y': 387.20000000000016, 'food_x': 116, 'food_y': 522, 'snake_body': [0.0, 10.57110258089134, 19.5944140688368], 'snake_body_pos': [[339.20000000000005, 387.20000000000016], [339.10400000009594, 376.6293333334296], [337.37600000958639, 367.69066667628044]]}


# [119, 97, 100, 115, None]
