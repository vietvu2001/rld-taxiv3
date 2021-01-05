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


def path_based(env, num_mods):
    # Train agent in original environment
    agent = QAgent(env)
    agent.qlearn(600, render=False)
    cell_dict = cell_frequency(agent)
    wall_dict = wall_interference(agent)

    opt_seq = None
    opt_val = float("-inf")
    
    for k in range(num_mods):
        seq = []
        ref = copy.deepcopy(env)
        # Pick k modifications on walls
        walls_to_remove = [wall_dict[i][0] for i in range(k)]
        for wall in walls_to_remove:
            ref = ref.transition([(wall)])
            seq.append((0, wall[0], wall[1]))

        num_special_cells = num_mods - k
        cells_to_assign = [cell_dict[i][0] for i in range(num_special_cells)]

        for cell in cells_to_assign:
            ref.special.append(cell)
            seq.append((1, cell[0], cell[1]))
        
        agent_k = QAgent(ref)
        print(colored("Iteration {} begins!".format(k), "red"))
        print(ref.walls, ref.special)
        agent_k.qlearn(600, render=False)

        rews = utility(agent_k)
        if rews > opt_val:
            opt_val = rews
            opt_seq = seq

    return (opt_seq, opt_val)

num_mods = 6
a = path_based(env, num_mods)
r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
file_dir = os.path.join(data_dir, "path_based_result_{}.txt".format(num_mods))

with open(file_dir, "w") as file:
    file.write("Modifications: ")
    file.write(str(a[0]))
    file.write("\n")
    file.write("Utility: ")
    file.write(str(a[1]))