# Test out the potency of the connected q-learning process

import numpy as np  # pylint: disable=import-error
import copy
import time
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
modified = copy.deepcopy(env)

# Reference environment

modified = modified.transition([(1, 4)])
modified.special.append((2, 1))

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

    new_agent = QAgent(modified)
    new_agent.qlearn(600, show=False, render=False)
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
    agent = QAgent(env)
    agent.qlearn(650)


    for iter in range(N):
        for i in range(num_processes):
            p = mp.Process(target=potency, args=(mutual, agent, modified, 400, i + iter * num_processes))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.terminate()

    print(np.mean(mutual))
    print(mutual)