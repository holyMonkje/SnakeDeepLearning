from ple.games.snake import Snake
from ple import PLE
import numpy as np
import time
import pandas as pd
from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# "snake_head_x","snake_head_y","food_x","food_y","action1","action1","action2","action3","action4","action5"

def predict_reward(snake_head_x,snake_head_y,food_x,food_y,action1,action2,action3,action4,action5):
	return loaded_model.predict(np.array([[snake_head_x,snake_head_y,food_x,food_y,action1,action2,action3,action4,action5]]))[0]


class NaiveAgent():
	"""
			This is our naive agent. It picks actions at random!
	"""

	def __init__(self, actions):
		self.actions = actions
		print(actions)

	
	def pickAction(self, reward, obs, state):
		
		predictions = []
		predictions.append(predict_reward(state["snake_head_x"],state["snake_head_y"],state["food_x"],state["food_y"],1,0,0,0,0))
		predictions.append(predict_reward(state["snake_head_x"],state["snake_head_y"],state["food_x"],state["food_y"],0,1,0,0,0))
		predictions.append(predict_reward(state["snake_head_x"],state["snake_head_y"],state["food_x"],state["food_y"],0,0,1,0,0))
		predictions.append(predict_reward(state["snake_head_x"],state["snake_head_y"],state["food_x"],state["food_y"],0,0,0,1,0))
		predictions.append(predict_reward(state["snake_head_x"],state["snake_head_y"],state["food_x"],state["food_y"],0,0,0,0,1))
		
		best_act = predictions.index(max(predictions))
		
		print(best_act)

		#do = self.actions[best_act]
		do = self.actions[np.random.randint(0, len(self.actions))]
		return do


game = Snake(width=640, height=640)
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

for i in range(learning_time):

	if i % 100 == 0:
		print(i/learning_time)

	if p.game_over():
		p.reset_game()

	#time.sleep(0.001)

	observation = p.getScreenRGB()
	state = p.game.getGameState()

	action = agent.pickAction(reward, observation,state)
	reward = p.act(action)

	actions.append(str(action))
	rewards.append(reward)	

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

actions_encoded = []

s = pd.Series(actions)
actions_encoded = pd.get_dummies(s)
print(s)

# [119, 97, 100, 115, None]
if '119' in actions_encoded:
	learning_sample['action1'] = actions_encoded['119']
else:
	learning_sample['action1'] = [0]*len(actions)

if '97' in actions_encoded:
	learning_sample['action2'] = actions_encoded['97']
else:
	learning_sample['action2'] = [0]*len(actions)

if '100' in actions_encoded:
	learning_sample['action3'] = actions_encoded['100']
else:
	learning_sample['action3'] = [0]*len(actions)

if '115' in actions_encoded:
	learning_sample['action4'] = actions_encoded['115']
else:
	learning_sample['action4'] = [0]*len(actions)

if 'None' in actions_encoded:
	learning_sample['action5'] = actions_encoded['None']
else:
	learning_sample['action5'] = [0]*len(actions)

learning_sample['reward'] = rewards

print(learning_sample)

learning_sample.to_csv("learning_sample.csv")