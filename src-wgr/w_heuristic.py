import csv
import ast
import numpy as np  # pylint: disable=import-error
import copy
import random
from collections import deque
from wgrenv import WindyGridworld
from w_qlearn import w_QAgent
from termcolor import colored  # pylint: disable=import-error

env = WindyGridworld()  # reference environment

def utility(agent):
    rewards = []
    starts = agent.env.resettable_states()
    for point in starts:
        r = agent.eval(fixed=point, show=False)[1]
        rewards.append(r)

    return np.mean(rewards)

# Notations
# (0, x, y): denote a jump cell
# (1, x, y): denote a diagonal (special) cell

def make_env(env, mod_seq):
    ref_env = copy.deepcopy(env)
    locations = mod_seq
    for element in locations:
        if element[0] == 0:
            ref_env.jump_cells.append((element[1], element[2]))
        else:
            ref_env.special.append((element[1], element[2]))
    
    return ref_env


def cell_frequency(agent):
    dict_return = {}
    for row in range(env.width):
        for col in range(env.length):
            if (row, col) not in agent.env.dest:
                dict_return[(row, col)] = 0

    ls = agent.env.resettable_states()
    for i in range(len(ls)):
        states = agent.eval(show=False, fixed=ls[i])[2]
        for state in states:
            if (state[0], state[1]) not in agent.env.dest:
                dict_return[(state[0], state[1])] += 1

    dict_return = sorted(dict_return.items(), key=lambda x: -x[1])
    return dict_return


if __name__ == "__main__":
    agent = w_QAgent(env)
    agent.qlearn(3000, render=False)
    cell_dict = cell_frequency(agent)
    for elem in cell_dict:
        print(elem)