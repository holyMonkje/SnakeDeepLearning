from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# load dataset
dataset = pd.read_csv("learning_sample.csv")
print(dataset)
# split into input (X) and output (Y) variables
X = dataset[["snake_head_x","snake_head_y","food_x","food_y","action"]].values
Y = dataset["reward"]

print(X)

model = Sequential()
# 12 neurons, 8 input vars
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=10, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))