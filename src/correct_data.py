import os

import numpy as np  # pylint: disable=import-error
import random
from gym import utils  # pylint: disable=import-error
from io import StringIO
import sys
from contextlib import closing
import copy

import gym  # pylint: disable=import-error
import time
from collections import deque
import csv
import ast

from sklearn.model_selection import train_test_split  # pylint: disable=import-error
from taxienv import TaxiEnv
from qlearn import QAgent
import matplotlib.pyplot as plt  # pylint: disable=import-error
from termcolor import colored  # pylint: disable=import-error
import multiprocessing as mp
from multiprocessing import Manager

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

r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
file_dir = os.path.join(data_dir, "data_trimmed_2.csv")

with open(file_dir, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        vector = ast.literal_eval(row['vector'])
        result = ast.literal_eval(row['average'])
        input_data.append(vector)
        output_data.append(result)


def make_env(env, input, data_index):
    ref_env = copy.deepcopy(env)
    locations = input[data_index]
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


def check(input_data, index, output_data):
    modified = make_env(env, input_data, index)
    agent = QAgent(modified)
    agent.qlearn(600, show=False)
    rews = utility(agent)
    return (rews == output_data[index])

for i in range(20):
    print(check(input_data, i, output_data))