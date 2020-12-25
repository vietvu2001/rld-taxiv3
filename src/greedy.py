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
from connect_qlearn import connected_qlearn

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
    # This function returns the sequence of modifications based on the wall and cell heuristics
    # Parameters
    # ===============================================================
    # env: the original environment
    # num_mods: the number of modifications
    # ===============================================================

    greedy_seq = []
    ref = copy.deepcopy(env)
    agent = None

    for i in range(num_mods):
        # For each iteration, find out the wall that most interferes and the cell that is crossed the most. Try out all options.
        if i == 0:
            agent = QAgent(ref)
            agent.qlearn(600, render=False)

        else:
            agent = connected_qlearn(agent, ref, 300)

        # Take out the lists from the heuristics.
        wall_dict = wall_interference(agent)
        cell_dict = cell_frequency(agent)
        

        # Take out the max values, and the options to try out.
        wall_nums = [elem[1] for elem in wall_dict]
        max_wall = max(wall_nums)

        cell_nums = [elem[1] for elem in cell_dict]
        max_cell = max(cell_nums)

        wall_options = [elem[0] for elem in wall_dict if elem[1] == max_wall]
        cell_options = [elem[0] for elem in cell_dict if elem[1] == max_cell]

        # Test out all the options, get optimal modification
        opt_value = float("-inf")
        opt_choice = None
        category = -1

        for wall in wall_options:
            print(colored("Testing environment", "red"))
            e = ref.transition([wall])
            new_agent = connected_qlearn(agent, e, 300)

            # Get utility
            val = utility(new_agent)
            if val > opt_value:
                opt_value = val
                opt_choice = wall
                category = 0

        for cell in cell_options:
            print(colored("Testing environment", "red"))
            e = copy.deepcopy(ref)
            e.special.append(cell)
            new_agent = connected_qlearn(agent, e, 300)

            # Get utility
            val = utility(new_agent)
            if val > opt_value:
                opt_value = val
                opt_choice = cell
                category = 1

        assert(category != -1)

        # Store found modification and change the reference environment
        if category == 0:
            mod = (0, opt_choice[0], opt_choice[1])
            greedy_seq.append(mod)
            ref = ref.transition([opt_choice])

        elif category == 1:
            mod = (1, opt_choice[0], opt_choice[1])
            greedy_seq.append(mod)
            ref.special.append(opt_choice)


    # Evaluate utility
    total_agent = QAgent(ref)
    total_agent.qlearn(600, render=False)
    result = utility(total_agent)
    print(colored(result, "red"))

    return greedy_seq, result


def greedy_iter(env, num_mods, num_trials):
    opt_val = float("-inf")
    opt_seq = None

    for i in range(num_trials):
        print(colored("Iteration {}".format(i + 1), "red"))
        seq, val = greedy(env, num_mods)
        if val > opt_val:
            opt_val = val
            opt_seq = seq

    return opt_seq, opt_val

# Test the greedy algorithm
num_mods = 6
mod_seq, value = greedy_iter(env, num_mods, 30)

# Store results
r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
txt_dir = os.path.join(data_dir, "greedy_{}.txt".format(num_mods))

with open(txt_dir, "w") as file:
    file.write("Modifications: ")
    file.write(str(mod_seq))
    file.write("\n")
    file.write("Utility: ")
    file.write(str(value))