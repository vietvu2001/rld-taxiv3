from w_mcts_trimmed import Node, Tree
from wgrenv import WindyGridworld
from w_qlearn import w_QAgent
from w_heuristic import utility
import numpy as np  # pylint: disable=import-error
from termcolor import colored  # pylint: disable=import-error
import copy
import time
import os

env = WindyGridworld()
num_iters = [120, 500, 1000, 2000, 2500, 3000]


def batch_greedy(env, num_mods, num_mods_per_run, ls_num_iters):
    # Parameters
    # =========================================================
    # env: original environment
    # num_mods: total number of modifications
    # num_mods_per_run: total number of modifications considered in combination
    # ls_num_iters: list of number of iterations run for each number of modifications
    # =========================================================

    # Example: [50, 200] means that if num_mods == 1, run 50 iterations, and if num_mods == 2, run 200 iterations.

    ref = copy.deepcopy(env)
    mods_ret = []  # answer of this algorithm

    # Initialize an MCTS tree
    tree = Tree(env, max_layer=num_mods)
    tree.initialize()

    # Keep a running count
    count = num_mods

    # Keep a running list of modifications
    ls_mods = copy.deepcopy(tree.modifications)

    # Initialize baseline
    baseline = 10.12

    assert(count >= num_mods_per_run)
    assert(len(ls_num_iters) == num_mods_per_run)

    while count > 0:
        print(colored(ls_mods, "red"))
        n = 0
        if count >= num_mods_per_run:
            n = num_mods_per_run

        else:
            n = count

        tree = Tree(ref, max_layer=n)
        tree.initialize()
        tree.threshold = baseline

        tree.modifications = copy.deepcopy(ls_mods)

        # Find out number of iterations
        num_iter = ls_num_iters[n - 1]

        # Perform an MCTS search
        tree.ucb_search(iterations=num_iter)

        a = tree.best_observed_choice()
        for elem in a[0]:
            mods_ret.append(elem)

        # Transform the environment
        for elem in a[0]:
            if elem[0] == 0:  # wall
                ref.jump_cells.append((elem[1], elem[2]))
            
            elif elem[0] == 1:  # cell
                ref.special.append((elem[1], elem[2]))

            ls_mods.remove(elem)

        count -= n

        # Increase baseline
        baseline += 0.4 * n


    # Find utility
    agent = w_QAgent(ref)
    agent.qlearn(3000)
    rews = utility(agent)

    return (mods_ret, rews)


num_mods = 4
num_mods_per_run = 1
ls = num_iters[0 : num_mods_per_run]
ans = batch_greedy(env, num_mods, num_mods_per_run, ls)

r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data-wgr")
txt_dir = os.path.join(data_dir, "batch_result_{}({}).txt".format(num_mods, num_mods_per_run))


with open(txt_dir, "w") as file:
    file.write("Modifications: ")
    file.write(str(ans[0]))
    file.write("\n")
    file.write("Utility: ")
    if ans[1] is not None:
        file.write(str(ans[1]))
    else:
        file.write("Utility not available")