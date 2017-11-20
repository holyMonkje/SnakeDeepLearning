from ple.games.snake import Snake
from ple import PLE
import numpy as np
import time
import pandas as pd

class NaiveAgent():
	"""
			This is our naive agent. It picks actions at random!
	"""

	def __init__(self, actions):
		self.actions = actions
		print(actions)

	def pickAction(self, reward, obs):
		do = self.actions[np.random.randint(0, len(self.actions))]
		return do


game = Snake(width=640, height=640)
p = PLE(game, fps=30, display_screen=False)
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

for i in range(learning_time):
	if p.game_over():
		p.reset_game()

	#time.sleep(0.001)

	observation = p.getScreenRGB()

	action = agent.pickAction(reward, observation)
	reward = p.act(action)

	actions.append(action)
	rewards.append(reward)

	state = p.game.getGameState()

	snake_head_x.append(state['snake_head_x'])
	snake_head_y.append(state['snake_head_y'])

	food_x.append(state['food_x'])
	food_y.append(state['food_y'])
	
	#print(state)

	if not reward == 0:
		print(reward)

	# {'snake_head_x': 339.20000000000005, 'snake_head_y': 387.20000000000016, 'food_x': 116, 'food_y': 522, 'snake_body': [0.0, 10.57110258089134, 19.5944140688368], 'snake_body_pos': [[339.20000000000005, 387.20000000000016], [339.10400000009594, 376.6293333334296], [337.37600000958639, 367.69066667628044]]}

learning_sample['food_x'] = food_x
learning_sample['food_y'] = food_y

learning_sample['snake_head_x'] = snake_head_x
learning_sample['snake_head_y'] = snake_head_y

learning_sample['action'] = actions
learning_sample['reward'] = rewards


print(learning_sample)

learning_sample.to_csv("learning_sample.csv")