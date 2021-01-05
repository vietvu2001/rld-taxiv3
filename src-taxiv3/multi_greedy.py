import os
import csv
import ast
import numpy as np  # pylint: disable=import-error
import copy
import random
import time
from collections import deque
from taxienv import TaxiEnv
from qlearn import QAgent
from termcolor import colored  # pylint: disable=import-error
from heuristic import cell_frequency, wall_interference, utility
from connect_qlearn import connected_qlearn
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
    # print(colored(result, "red"))

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


def multi_greedy(mutual_info, env, num_mods, index):
    # Multiprocessing version of greedy algorithm to boost time
    # Parameters
    # ======================================================
    # mutual_info: a list of results up to run
    # env, num_mods: belonging to the greedy algorithm
    # index: the position that upon ending, this function will put the result into mutual_info
    # ======================================================

    seq, res = greedy(env, num_mods)
    mutual_info[index] = (seq, res)

    return 

if __name__ == "__main__":
    num_mods = 6

    processes = []
    mp.set_start_method = "spawn"
    num_processes = 4
    manager = Manager()
    mutual = manager.list()

    N = 15
    for _ in range(N * num_processes):
        mutual.append(0)

    start = time.time()
    for iter in range(N):
        for i in range(num_processes):
            p = mp.Process(target=multi_greedy, args=(mutual, env, num_mods, i + iter * num_processes))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.terminate()

    end = time.time()
    print(end - start)

    vals = [elem[1] for elem in mutual]
    opt_index = np.argmax(vals)

    # Store results
    r_dir = os.path.abspath(os.pardir)
    data_dir = os.path.join(r_dir, "data")
    txt_dir = os.path.join(data_dir, "multi_greedy_{}.txt".format(num_mods))

    with open(txt_dir, "w") as file:
        file.write("Modifications: ")
        file.write(str(mutual[opt_index][0]))
        file.write("\n")
        file.write("Utility: ")
        file.write(str(mutual[opt_index][1]))

     