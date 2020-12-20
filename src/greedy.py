import os
import csv
import ast
import numpy as np  # pylint: disable=import-error
import copy
import random
from collections import deque
from taxienv import TaxiEnv
from qlearn import QAgent
from termcolor import colored  # pylint: disable=import-error
from heuristic import cell_frequency, wall_interference, utility

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

def greedy(env, num_mods):
    result = []
    ref_env = copy.deepcopy(env)
    total = 0
    for _ in range(num_mods):
        agent = QAgent(ref_env)
        agent.qlearn(600, render=False)
        wall_dict = wall_interference(agent)
        cell_dict = cell_frequency(agent)

        first = None
        second = None
        first_type = None
        second_type = None
        modified_1 = None
        modified_2 = None

        if len(wall_dict) != 0 and len(cell_dict) != 0:
            first = wall_dict[0][0]
            second = cell_dict[0][0]
            first_type = 0
            second_type = 1
            modified_1 = ref_env.transition([first])
            modified_2 = copy.deepcopy(ref_env)
            modified_2.special.append(second)

        elif len(wall_dict) == 0:
            first = cell_dict[0][0]
            second = cell_dict[1][0]
            first_type = 1
            second_type = 1
            modified_1 = copy.deepcopy(ref_env)
            modified_2 = copy.deepcopy(ref_env)
            modified_1.special.append(first)
            modified_2.special.append(second)

        agent_1 = QAgent(modified_1)
        agent_2=  QAgent(modified_2)
        agent_1.qlearn(600, render=False)
        agent_2.qlearn(600, render=False)

        u_1 = utility(agent_1)
        u_2 = utility(agent_2)

        if u_1 >= u_2:
            ref_env = copy.deepcopy(modified_1)
            result.append((first_type, first[0], first[1]))
            total = u_1

        else:
            ref_env = copy.deepcopy(modified_2)
            result.append((second_type, second[0], second[1]))
            total = u_2
    
    return (result, total)

print(greedy(env, 6))