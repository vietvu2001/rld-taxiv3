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
from taxienv import TaxiEnv
from qlearn import QAgent
from termcolor import colored  # pylint: disable=import-error

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
num_mods = 6  # specify here

r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
file_dir = os.path.join(data_dir, "connected_data_trimmed_{}.csv".format(num_mods))

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
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss = 'mae')

# Training
TEST_SIZE = 0.2
x_train, x_test, y_train, y_test = train_test_split(np.array(input_data), np.array(output_data), test_size=TEST_SIZE)

model.fit(x_train, y_train, epochs=600)
model.evaluate(x_test, y_test, verbose=2)


class Elem():
    def __init__(self, seq, val):
        self.seq = seq
        self.val = val


class Heap():
    def __init__(self, model, data, size):
        self.array = deque(maxlen=size)
        np_data = np.array(data)
        val_vector = model.predict(np_data)

        for i in range(len(data)):
            element = np.reshape(data[i], (1, num_mods * 3))
            value = val_vector[i][0]

            e = Elem(element, value)

            self.array.append(e)

        self.model = model
        self.size = size

        assert(len(data) == self.size)

        # Now the heap's array contains instances of Elem


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

        # if l < len(self.array) and self.model.predict(self.array[l])[0][0] < self.model.predict(self.array[N])[0][0]:
        if l < len(self.array) and self.array[l].val < self.array[N].val:
            largest = l

        else:
            largest = N


        # if r < len(self.array) and self.model.predict(self.array[r])[0][0] < self.model.predict(self.array[largest])[0][0]:
        if r < len(self.array) and self.array[r].val < self.array[largest].val:
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
            # val_element = self.model.predict(self.array[i])[0][0]
            val_element = self.array[i].val
            # val_parent = self.model.predict(self.array[self.parent(i)])[0][0]
            val_parent = self.array[self.parent(i)].val
            if val_parent > val_element:
                return False
        
        return True


    def extract_min(self):
        min_seq = copy.deepcopy(self.array[0].seq)
        min_val = copy.deepcopy(self.array[0].val)

        self.array[0] = copy.deepcopy(self.array[len(self.array) - 1])
        self.array.pop()
        self.min_heapify(0)

        return (min_seq, min_val)

    
    def peek(self):
        # return self.model.predict(self.array[0])[0][0]
        return self.array[0].val


    def value_list(self):
        check = []
        for i in range(len(self.array)):
            # check.append(self.model.predict(self.array[i])[0][0])
            check.append(self.array[i].val)

        return check

    
    def insert(self, mods, val):

        self.extract_min()
        e = Elem(mods, val)
        self.array.append(e)
        
        N = len(self.array) - 1
        if not N == self.size - 1:
            print(N)
            assert(N == self.size - 1)

        # while N > 0 and self.model.predict(self.array[self.parent(N)])[0][0] > self.model.predict(self.array[N])[0][0]:
        while N > 0 and self.array[self.parent(N)].val > self.array[N].val:
            temp = copy.deepcopy(self.array[self.parent(N)])
            self.array[self.parent(N)] = copy.deepcopy(self.array[N])
            self.array[N] = copy.deepcopy(temp)

            N = self.parent(N)

        # assert(self.verify_min_heap())
        assert(len(self.array) == self.size)

    
    def mod_seq(self, index):
        a = copy.deepcopy(self.array[index].seq)
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


modifications = []
for wall in env.walls:
    modifications.append((0, wall[0], wall[1]))
for row in range(env.width):
    for col in range(env.length):
        modifications.append((1, row, col))



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

N = int(scipy.special.binom(len(modifications), num_mods))


if num_mods == 6:
    num_trials = int(2e+6)

else:
    num_trials = int(1e+6)

# Initialize and build heap
sz = min(12 * num_mods, len(x_test))
h = Heap(model, x_test[0 : sz], sz)
h.build_heap()


ls = []
for i in range(num_trials):
    seq = get_ordered_sequence(modifications, k=num_mods)
    seq = np.array(seq)
    seq = np.reshape(seq, (num_mods * 3))

    ls.append(seq)

ls = np.array(ls)
vector = model.predict(ls)


for i in range(num_trials):
    val = vector[i][0]
    if val > h.peek():
        checkls = [elem.seq for elem in h.array]
        seq = np.reshape(ls[i], (1, num_mods * 3))
        exist = False

        for s in checkls:
            b = s == seq
            if b.all() == True:
                exist = True
                break
        
        if not exist:
            h.insert(seq, val)
    
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
    print(colored(rews, "red"))
    if rews > opt_val:
        opt_val = rews
        opt_seq = seq

r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
file_dir = os.path.join(data_dir, "connected_sl_trimmed_result_{}.txt".format(num_mods))
with open(file_dir, "w") as file:
    file.write("Modifications: ")
    file.write(str(opt_seq))
    file.write("\n")
    file.write("Utility: ")
    file.write(str(opt_val))