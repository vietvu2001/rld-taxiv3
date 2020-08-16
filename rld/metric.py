import numpy as np  # pylint: disable=import-error
import random
from gym import utils  # pylint: disable=import-error
from io import StringIO
import sys
from contextlib import closing
import copy

import gym  # pylint: disable=import-error
import time
import random
from collections import deque

import tensorflow as tf  # pylint: disable=import-error
from tensorflow import keras  # pylint: disable=import-error
from tensorflow.keras import layers  # pylint: disable=import-error
from keras.layers import Reshape, BatchNormalization  # pylint: disable=import-error
from keras.layers.embeddings import Embedding  # pylint: disable=import-error

from taxi_design import evaluate_branch
from taxienv import TaxiEnv
from total_special import evaluate

map = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

gamma = 1  # Discount factor for past rewards
max_steps_per_episode = 3000
eps = 1e-6

map_to_numpy = np.asarray(map, dtype='c')
env = TaxiEnv(map_to_numpy)

env_modified = env.transition([(4, 2), (2, 4)])
env_modified.special.append((2, 1))
env_modified.special.append((2, 2))

env_modified_1 = env
env_modified_1.special.append((2, 1))
env_modified_1.special.append((2, 2))
env_modified_1.special.append((2, 3))

actor = keras.models.load_model("my_actor")
critic = keras.models.load_model("my_critic")
actor_modified = keras.models.load_model("my_actor_freq_2")
critic_modified = keras.models.load_model("my_critic_freq_2")
actor_modified_1 = keras.models.load_model("my_actor_freq_1")
critic_modified_1 = keras.models.load_model("my_critic_freq_1")

metric = [0] * 3
index = {0, 1, 2}

for _ in range(500):
    point = env.reset()
    env_modified.reset()
    env_modified_1.reset()
    env_modified.current = point
    env_modified_1.current = point
    a = len(evaluate(actor, critic, env, point, show=False)[0])
    b = len(evaluate_branch(actor_modified, critic_modified, env_modified, point, show=False)[0])
    c = len(evaluate_branch(actor_modified_1, critic_modified_1, env_modified_1, point, show=False)[0])
    metric[0] += a
    metric[1] += b
    metric[2] += c

for i in range(len(metric)):
    metric[i] /= 500
print(metric)


