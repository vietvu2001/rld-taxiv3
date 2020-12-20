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
ref = copy.deepcopy(env)

# Reference environment

ref = ref.transition([(1, 4), (2, 4)])
ref.special.append((3, 3))
ref.special.append((4, 2))

modified = ref.transition([(5, 6)])

# Design experiment
N = 100

def potency(agent, modified, num_episodes):
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

    a = np.linalg.norm(l_vals - n_vals)
    print(a)


if __name__ == "__main__":
    processes = []
    mp.set_start_method = "spawn"
    num_processes = 10

    
    # Create default agent
    agent = QAgent(ref)
    agent.qlearn(650)


    for iter in range(N):
        for i in range(num_processes):
            p = mp.Process(target=potency, args=(agent, modified, 300))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.terminate()