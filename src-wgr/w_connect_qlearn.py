import os
import csv
import ast
import numpy as np  # pylint: disable=import-error
import copy
import random
import time
from collections import deque
from wgrenv import WindyGridworld
from w_qlearn import w_QAgent
from termcolor import colored  # pylint: disable=import-error
from w_heuristic import cell_frequency, utility


def connected_qlearn(agent, new_env, num_episodes):
    # Parameters
    # ==============================================
    # agent: pre-trained agent in some environment
    # new_env: new environment
    # ==============================================

    # We will use the pre-trained agent to train it in the new environment
    # Intuition is that the q-values only need slight changes, so it will be computationally wasteful to calculate from scratch

    linked_agent = w_QAgent(new_env)
    linked_agent.q = copy.deepcopy(agent.q)  # linking the q-values together

    linked_agent.epsilon = 0.75
    linked_agent.qlearn(num_episodes, render=False)

    return linked_agent