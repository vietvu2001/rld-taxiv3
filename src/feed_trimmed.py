import os

import csv
import ast
import numpy as np  # pylint: disable=import-error
import copy
import random
import math
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
from taxienv import TaxiEnv
from qlearn import QAgent
from termcolor import colored  # pylint: disable=import-error
from heuristic import wall_interference, cell_frequency

map = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

map_to_numpy = np.asarray(map, dtype="c")
env = TaxiEnv(map_to_numpy)  # reference environment

input_data = []
output_data = []
num_mods = 5  # specify here

r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
file_dir = os.path.join(data_dir, "data_trimmed_{}.csv".format(num_mods))

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
model.add(Dense(64, input_shape=(num_mods * 3, ), activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.002), loss = 'mae')

# Training
TEST_SIZE = 0.2
x_train, x_test, y_train, y_test = train_test_split(np.array(input_data), np.array(output_data), test_size=TEST_SIZE)

model.fit(x_train, y_train, epochs=400)
model.evaluate(x_test, y_test, verbose=2)

class Heap():
    def __init__(self, model, data, size):
        self.array = deque(maxlen=size)
        for element in data:
            element = np.reshape(element, (1, num_mods * 3))
            self.array.append(element)

        self.model = model
        self.size = size


    def parent(self, i):
        if i % 2 != 0:
            return i // 2
        else:
            return i // 2 - 1


    def left(self, i):
        return i * 2 + 1


    def right(self, i):
        return i * 2 + 2


    def min_heapify(self, N):
        l = self.left(N)
        r = self.right(N)

        if l < len(self.array) and self.model.predict(self.array[l])[0][0] < self.model.predict(self.array[N])[0][0]:
            largest = l

        else:
            largest = N

        if r < len(self.array) and self.model.predict(self.array[r])[0][0] < self.model.predict(self.array[largest])[0][0]:
            largest = r

        if largest != N:
            temp = copy.deepcopy(self.array[N])
            self.array[N] = copy.deepcopy(self.array[largest]) 
            self.array[largest] = copy.deepcopy(temp)
            self.min_heapify(largest)


    def build_heap(self):
        for i in range(len(self.array) // 2 - 1, -1, -1):
            self.min_heapify(i)
        
        assert(self.verify_min_heap())

    
    def verify_min_heap(self):
        for i in range(1, len(self.array)):
            val_element = self.model.predict(self.array[i])[0][0]
            val_parent = self.model.predict(self.array[self.parent(i)])[0][0]
            if val_parent > val_element:
                return False
        
        return True


    def extract_min(self):
        min_seq = self.array[0]
        self.array[0] = copy.deepcopy(self.array[len(self.array) - 1])
        self.array.pop()
        self.min_heapify(0)

        return (min_seq, self.model.predict(min_seq)[0][0])

    
    def peek(self):
        return self.model.predict(self.array[0])[0][0]


    def value_list(self):
        check = []
        for i in range(len(self.array)):
            check.append(self.model.predict(self.array[i])[0][0])

        return check

    
    def insert(self, mods):
        assert(mods.shape == (1, num_mods * 3))  
        value = self.model.predict(mods)[0][0]
        assert(self.peek() < value)

        self.extract_min()
        self.array.append(mods)
        N = len(self.array) - 1
        if not N == self.size - 1:
            print(N)
            assert(N == self.size - 1)

        while N > 0 and self.model.predict(self.array[self.parent(N)])[0][0] > self.model.predict(self.array[N])[0][0]:
            temp = copy.deepcopy(self.array[self.parent(N)])
            self.array[self.parent(N)] = copy.deepcopy(self.array[N])
            self.array[N] = copy.deepcopy(temp)

            N = self.parent(N)

        assert(self.verify_min_heap())
        assert(len(self.array) == self.size)

    
    def mod_seq(self, index):
        a = copy.deepcopy(self.array[index])
        a = list(a[0])
        seq = []
        for i in range(0, num_mods * 3, 3):
            mod = tuple(a[i : (i + 3)])
            seq.append(mod)
        
        return seq


def make_env(env, mod_seq):
    ref_env = copy.deepcopy(env)
    locations = mod_seq
    for element in locations:
        if element[0] == 0:
            ref_env = ref_env.transition([(element[1], element[2])])
        else:
            ref_env.special.append((element[1], element[2]))
    
    return ref_env

def utility(agent):
    rewards = []
    starts = agent.env.resettable_states()
    for point in starts:
        r = agent.eval(fixed=point, show=False)[1]
        rewards.append(r)

    return np.mean(rewards)


agent = QAgent(env)
agent.qlearn(600, show=False)
cell_dict = cell_frequency(agent)
wall_dict = wall_interference(agent)
modifications = []

for element in wall_dict:
    modifications.append((0, element[0][0], element[0][1]))
for element in cell_dict[0 : 14]:
    row, col = element[0]
    modifications.append((1, row, col))


h = Heap(model, x_test[0 : min(8 * num_mods, len(x_test))], min(8 * num_mods, len(x_test)))
h.build_heap()


def get_ordered_sequence(list, k):
    N = len(list)
    res = []
    index = 0
    for i in range(k):
        if i == 0:
            index = random.randint(0, N - k)
        else:
            index = random.randint(index + 1, N - k + i)

        res.append(list[index])
        
    return res


if num_mods == 6:
    num_trials = min(int(scipy.special.binom(len(modifications), num_mods)), 100000)

else:
    num_trials = min(int(scipy.special.binom(len(modifications), num_mods)), 50000)


for i in range(num_trials):
    #seq = random.sample(modifications, k=4)
    seq = get_ordered_sequence(modifications, k=num_mods)

    seq = np.array(seq)
    seq = np.reshape(seq, (1, num_mods * 3))
    val = model.predict(seq)[0][0]
    if val > h.peek():
        h.insert(seq)
    
    if i % 100 == 0:
        print(i)

opt_seq = None
opt_val = -1
for element in range(len(h.array)):
    seq = h.mod_seq(element)
    modified = make_env(env, seq)
    agent = QAgent(modified)
    agent.qlearn(600, render=False)
    rews = utility(agent)
    if rews > opt_val:
        opt_val = rews
        opt_seq = seq

r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
file_dir = os.path.join(data_dir, "sl_trimmed_result_{}.txt".format(num_mods))
with open(file_dir, "w") as file:
    file.write("Modifications: ")
    file.write(str(opt_seq))
    file.write("\n")
    file.write("Utility: ")
    file.write(str(opt_val))