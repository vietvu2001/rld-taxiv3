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

modifications = []
for wall in env.walls:
    modifications.append((0, wall))
for row in range(env.width):
    for col in range(env.length):
        modifications.append((1, (row, col)))


def simulate_env(env, num_changes):
    features = []
    mods = random.sample(modifications, k=num_changes)
    result = copy.deepcopy(env)

    for change in mods:
        if change[0] == 0:  # walls
            result = result.transition([change[1]])
        elif change[0] == 1:  # cells
            result.special.append(change[1])
        features.append((change[0], change[1][0], change[1][1]))

    features = sorted(features)
    return [result, features] 


def utility(agent):
    rewards = []
    starts = agent.env.resettable_states()
    for point in starts:
        r = agent.eval(fixed=point, show=False)[1]
        rewards.append(r)

    return np.mean(rewards)


def qlearn_as_func(agent, env, number, agents, insert_position=-1):
    agent.qlearn(600, number=number, render=False)
    agents[insert_position] = agent


data = []

if __name__ == "__main__":
    rounds = 800
    mp.set_start_method = "spawn"
    num_processes = 10
    processes = []
    manager = Manager()
    agents = manager.list()
    for i in range(rounds * num_processes):
        agents.append(0)  # keeper

    categories = []
    num_mods = 6

    for iter in range(rounds):
        print(colored("Data addition round {} begins!".format(iter), "red"))
        for i in range(num_processes):
            results = simulate_env(env, num_mods)
            modified = results[0]
            categories.append(results[1])
            agent = QAgent(modified)
            p = mp.Process(target=qlearn_as_func, args=(agent, modified, i, agents, i + iter * num_processes))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.terminate()

    
    for i in range(len(agents)):
        ut = utility(agents[i])
        data.append((categories[i], ut))

    r_dir = os.path.abspath(os.pardir)
    data_dir = os.path.join(r_dir, "data")
    file_dir = os.path.join(data_dir, "data_{}.csv".format(num_mods))

    with open(file_dir, "w") as file:
        fieldnames = ["vector", "average"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for element in data:
            simple_dict = {}
            simple_dict["vector"] = element[0]
            simple_dict["average"] = element[1]
            writer.writerow(simple_dict)
    
