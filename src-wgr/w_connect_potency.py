# Test out the potency of the connected q-learning process

import numpy as np  # pylint: disable=import-error
import copy
import time
from wgrenv import WindyGridworld
from w_qlearn import w_QAgent
from termcolor import colored  # pylint: disable=import-error
from w_heuristic import cell_frequency, utility
from w_connect_qlearn import connected_qlearn
import multiprocessing as mp
from multiprocessing import Manager

env = WindyGridworld()  # reference environment
modified = copy.deepcopy(env)

# Reference environment

modified.jump_cells.append((4, 1))
modified.jump_cells.append((4, 5))
modified.special.append((1, 6))
modified.special.append((3, 4))

# Design experiment
N = 10

def potency(mutual_ls, agent, modified, num_episodes, index):
    # This function tests the potency of the connected q-learning paradigm.
    # Parameters
    # ==============================================
    # agent: pre-trained agent in some environment
    # modified: new environment
    # num_episodes: number of episodes trained in connected q-learning paradigm
    # ==============================================

    series = agent.env.resettable_states()

    conn_agent = connected_qlearn(agent, modified, num_episodes)
    l_vals = []

    for state in series:
        res = conn_agent.eval(fixed=state, show=False)[1]
        l_vals.append(res)

    new_agent = w_QAgent(modified)
    new_agent.qlearn(3500, show=False, render=False)
    n_vals = []

    for state in series:
        res = new_agent.eval(fixed=state, show=False)[1]
        n_vals.append(res)

    l_vals = np.array(l_vals)
    n_vals = np.array(n_vals)

    a = abs(np.sum(l_vals) - np.sum(n_vals))
    
    mutual_ls[index] = a


if __name__ == "__main__":
    processes = []
    mp.set_start_method = "spawn"
    num_processes = 10
    manager = Manager()
    mutual = manager.list()

    for _ in range(N * num_processes):
        mutual.append(0)
    
    # Create default agent
    agent = w_QAgent(env)
    agent.qlearn(3000)


    for iter in range(N):
        for i in range(num_processes):
            p = mp.Process(target=potency, args=(mutual, agent, modified, 3000, i + iter * num_processes))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.terminate()

    print(np.mean(mutual))
    print(mutual)