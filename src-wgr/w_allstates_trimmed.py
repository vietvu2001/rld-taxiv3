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
from wgrenv import WindyGridworld
from w_qlearn import w_QAgent
import matplotlib.pyplot as plt  # pylint: disable=import-error
from termcolor import colored  # pylint: disable=import-error
import multiprocessing as mp
from multiprocessing import Manager
from w_heuristic import cell_frequency
import itertools


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


def qlearn_as_func(agent, env, number, agents, insert_position=-1):
    agent.qlearn(3000, number=number, render=False)
    agents[insert_position] = agent


data = []

if __name__ == "__main__":
    rounds = 300
    mp.set_start_method = "spawn"
    num_processes = 10
    processes = []
    manager = Manager()
    agents = manager.list()
    for i in range(rounds * num_processes):
        agents.append(0)  # keeper

    categories = []
    num_mods = 4

    map_to_numpy = np.asarray(map, dtype="c")
    env = WindyGridworld()  # reference environment

    orig_agent = w_QAgent(env)
    orig_agent.qlearn(3000, render=False)
    cell_dict = cell_frequency(orig_agent)
    modifications = []

    for element in cell_dict:
        if element[1] != 0:
            modifications.append((0, element[0][0], element[0][1]))
            modifications.append((1, element[0][0], element[0][1]))

    modifications.sort()
    ls = None

    if num_mods == 1:
        ls = [[elem] for elem in modifications]

    else:
        ls = list(itertools.combinations(modifications, num_mods))

    if num_mods > 1:
        for i in range(len(ls)):
            ls[i] = list(ls[i])
            ls[i].sort()

    random.shuffle(ls)

    if num_mods == 1:
        chosen_vectors = ls

    else:
        chosen_vectors = ls[0 : (rounds * num_processes)]


    for iter in range(rounds):
        print(colored("Data addition round {} begins!".format(iter), "red"))
        for i in range(num_processes):
            if i + iter * num_processes >= len(chosen_vectors):
                break

            # results = simulate_env(env, num_mods)
            v = chosen_vectors[i + iter * num_processes]
            # modified = results[0]
            modified = make_env(env, v)
            # categories.append(results[1])
            categories.append(v)

            agent = w_QAgent(modified)

            p = mp.Process(target=qlearn_as_func, args=(agent, modified, i, agents, i + iter * num_processes))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.terminate()

    for i in range(len(agents)):
        if agents[i] != 0:
            ut = utility(agents[i])
            data.append((categories[i], ut))

    r_dir = os.path.abspath(os.pardir)
    data_dir = os.path.join(r_dir, "data-wgr")
    file_dir = os.path.join(data_dir, "data_trimmed_{}.csv".format(num_mods))

    with open(file_dir, "w") as file:
        fieldnames = ["vector", "average"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for element in data:
            simple_dict = {}
            simple_dict["vector"] = element[0]
            simple_dict["average"] = element[1]
            writer.writerow(simple_dict)