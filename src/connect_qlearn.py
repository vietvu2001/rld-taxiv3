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

map = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


def connected_qlearn(agent, new_env, num_episodes):
    # Parameters
    # ==============================================
    # agent: pre-trained agent in some environment
    # new_env: new environment
    # ==============================================

    # We will use the pre-trained agent to train it in the new environment
    # Intuition is that the q-values only need slight changes, so it will be computationally wasteful to calculate from scratch

    linked_agent = QAgent(new_env)
    linked_agent.q = copy.deepcopy(agent.q)  # linking the q-values together

    linked_agent.epsilon = 0.75
    linked_agent.qlearn(num_episodes, render=False)

    return linked_agent


'''map_to_numpy = np.asarray(map, dtype="c")
env = TaxiEnv(map_to_numpy)  # reference environment
ref = copy.deepcopy(env)

# Reference environment

ref = ref.transition([(1, 4), (2, 4)])
ref.special.append((3, 3))
ref.special.append((4, 2))

modified = ref.transition([(5, 6)])

agent = QAgent(ref)
agent.qlearn(700)

start = time.time()
linked_agent = connected_qlearn(agent, modified, 250)
end = time.time()

print(end - start)'''