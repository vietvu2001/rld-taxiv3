import os

import csv
import ast
import numpy as np  # pylint: disable=import-error
import copy
import random
import math
import time
import scipy.special  #pylint: disable=import-error

from sklearn.model_selection import train_test_split  # pylint: disable=import-error
import tensorflow as tf  # pylint: disable=import-error
from tensorflow import keras  # pylint: disable=import-error
from tensorflow.keras import layers  # pylint: disable=import-error
from keras.layers import Reshape, BatchNormalization, Dense, Dropout, Embedding  # pylint: disable=import-error
from keras.layers.embeddings import Embedding  # pylint: disable=import-error
from keras.models import Sequential  # pylint: disable=import-error
from keras.optimizers import Adam  # pylint: disable=import-error
from keras.losses import MeanAbsoluteError  # pylint: disable=import-error
from collections import deque
from wgrenv import WindyGridworld
from w_qlearn import w_QAgent
from termcolor import colored  # pylint: disable=import-error
import itertools

env = WindyGridworld()  # reference environment

input_data = []
output_data = []
num_mods = 4 # specify here

r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data-wgr")
file_dir = os.path.join(data_dir, "data_{}.csv".format(num_mods))

# Read data
with open(file_dir, "r") as f:
  reader = csv.DictReader(f)
  for row in reader:
    vector = ast.literal_eval(row['vector'])
    num_mods = len(vector)
    result = ast.literal_eval(row['average'])
    vector = np.array(vector)
    vector = np.reshape(vector, (num_mods * 3))
    input_data.append(vector)
    output_data.append(result)

shape_tuple = input_data[0].shape
print(num_mods == shape_tuple[0] // 3)

# Build model
model = Sequential()
model.add(Dense(32, input_shape=(num_mods * 3, ), activation="relu"))
model.add(Dense(128, activation="tanh"))
model.add(BatchNormalization())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.0002), loss = tf.keras.losses.Huber())

# Training
TEST_SIZE = 0.2
x_train, x_test, y_train, y_test = train_test_split(np.array(input_data), np.array(output_data), test_size=TEST_SIZE)

model.fit(x_train, y_train, epochs=400)
model.evaluate(x_test, y_test, verbose=2)

# Retrain with different split
TEST_SIZE = 0.2
x_train, x_test, y_train, y_test = train_test_split(np.array(input_data), np.array(output_data), test_size=TEST_SIZE)

model.fit(x_train, y_train, epochs=400)
model.evaluate(x_test, y_test, verbose=2)


def make_env(env, mod_seq):
    ref_env = copy.deepcopy(env)
    locations = mod_seq
    for element in locations:
        if element[0] == 0:
            ref_env.jump_cells.append((element[1], element[2]))
        else:
            ref_env.special.append((element[1], element[2]))
    
    return ref_env


def utility(agent):
    rewards = 0
    count = 0

    starts = agent.env.resettable_states()
    for point in starts:
        r = agent.eval(fixed=point, show=False)[1]
        rewards += r
        count += 1

    return rewards / count


modifications = []
for row in range(env.width):
    for col in range(env.length):
        modifications.append((0, row, col))

for row in range(env.width):
    for col in range(env.length):
        modifications.append((1, row, col))


ls = list(itertools.combinations(modifications, num_mods))[0 : 10]
for i in range(len(ls)):
    ls[i] = np.array(ls[i])
    ls[i] = np.reshape(ls[i], (num_mods * 3))

ls = np.array(ls)
vector = model.predict(ls)

# Keep track of max and corresponding environment
s = vector.shape
a = np.reshape(vector, (s[0] * s[1]))
index = np.argmax(a)

# Vector at index of highest prediction
corr_vec = ls[index]
coor_vec = list(np.reshape(corr_vec, (len(corr_vec) // 3, 3)))

res_env = make_env(env, coor_vec)
agent = w_QAgent(res_env)
agent.qlearn(3000, render=False)
opt_val = utility(agent)

# Re-format found vector
x = copy.deepcopy(coor_vec)
for i in range(len(x)):
    x[i] = tuple(x[i])


r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data-wgr")
file_dir = os.path.join(data_dir, "sl_nh_result_{}.txt".format(num_mods))
with open(file_dir, "w") as file:
    file.write("Modifications: ")
    file.write(str(x))
    file.write("\n")
    file.write("Utility: ")
    file.write(str(opt_val))
    file.write("\n")
    file.write("Number of iterations: {}".format(len(input_data)))