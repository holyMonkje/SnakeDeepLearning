from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# load dataset
dataset = pd.read_csv("learning_sample.csv")
print(dataset)
# split into input (X) and output (Y) variables

cols = ["snake_head_x","snake_head_y","food_x","food_y","action1","action2","action3","action4","action5"]

X = dataset[cols].values
Y = dataset["reward"]

print(Y)

model = Sequential()
# 12 neurons, 8 input vars
model.add(Dense(len(cols), input_dim=len(cols), activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='relu'))


# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=5, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")