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

map_to_numpy = np.asarray(map, dtype="c")
env = TaxiEnv(map_to_numpy)  # reference environment

# Changing environments
modified = env.transition([(1, 4), (5, 2), (5, 6)])
modified.special.append((1, 1))
modified.special.append((3, 3))

agent = QAgent(env)
agent.qlearn(700, show=False, render=False)


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

    linked_agent.epsilon = 0.75  # encouraging exploration
    linked_agent.qlearn(num_episodes, show=False, render=False)

    return linked_agent

# Trial run
start = time.time()
linked_agent = connected_qlearn(agent, modified, 200)
end = time.time()

series = modified.resettable_states()
l_vals = []

for state in series:
    res = linked_agent.eval(show=False, fixed=state)
    l_vals.append(res[1])

l_vals = np.array(l_vals)

print(end - start)