import numpy as np  # pylint: disable=import-error
import random
from gym import utils  # pylint: disable=import-error
from io import StringIO
import sys
from contextlib import closing
import copy

import gym  # pylint: disable=import-error
import time
from collections import deque


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


class TaxiEnv():
    def __init__(self, map_encode_by_numpy):
        self.length = 5
        self.width = 5
        self.max_row = self.width - 1
        self.max_col = self.length - 1

        # Destinations
        self.dest = [(0, 0), (0, 4), (4, 0), (4, 3)]

        # Number of states
        self.num_states = 500

        # numpy encoding of map
        self.desc = map_encode_by_numpy

        dims = self.desc.shape
        self.walls = []
        self.non_walls = []

        for i in range(1, dims[0] - 1):
            for j in range(1, dims[1] - 1):
                if self.desc[i][j] == b"|":
                    self.walls.append((i, j))
                elif self.desc[i][j] == b":":
                    self.non_walls.append((i, j))

        self.reward_cells = []
        self.current = None
        self.special = []


    def terminal(self, state):
        if state in self.dest:
            return True
        return False

    # Represent state: (row, col, pass_id, dest_id)
    # id: 0 - (0, 0), 1 - (0, 4), 2 - (4, 0), 3 - (4, 3)

    def step(self, action):

        assert self.current != None

        state = self.decode(self.current)

        taxi_loc = (state[0], state[1])
        new_row, new_col, new_pass_idx, dest_id = state
        done = False
        reward = -1

        if action == 0:
            new_row = min(state[0] + 1, self.max_row)
        elif action == 1:
            new_row = max(state[0] - 1, 0)
        elif action == 2 and self.desc[1 + state[0], 2 * state[1] + 2] == b":":
            new_col = min(state[1] + 1, self.max_col)
        elif action == 3 and self.desc[1 + state[0], 2 * state[1]] == b":":
            new_col = max(state[1] - 1, 0)
        elif action == 4:  # pickup
            if (state[2] < 4 and taxi_loc == self.dest[state[2]]):
                new_pass_idx = 4
            else: # passenger not at location
                reward = -10
        elif action == 5:  # dropoff
            if (taxi_loc == self.dest[state[3]]) and state[2] == 4:
                new_pass_idx = state[3]
                done = True
                reward = 20
            elif (taxi_loc in self.dest) and state[2] == 4:
                new_pass_idx = self.dest.index(taxi_loc)
            else: # dropoff at wrong location
                reward = -10
                
        elif len(self.special) != 0 and taxi_loc in self.special:
            if action == 6:
                propo = self.desc[state[0], 2 * state[1] + 2] == b":" or self.desc[state[0] + 1, 2 * state[1] + 2] == b":"
                if propo and state[0] >= 1 and state[1] + 1 <= self.max_col:
                    new_row = state[0] - 1
                    new_col = state[1] + 1

            elif action == 7:
                propo = self.desc[state[0] + 1, 2 * state[1] + 2] == b":" or self.desc[state[0] + 2, 2 * state[1] + 2] == b":"
                if propo and state[0] + 1 <= self.max_row and state[1] + 1 <= self.max_col:
                    new_row = state[0] + 1
                    new_col = state[1] + 1

            elif action == 8:
                propo = self.desc[state[0] + 1, 2 * state[1]] == b":" or self.desc[state[0] + 2, 2 * state[1]] == b":"
                if propo and state[0] + 1 <= self.max_row and state[1] >= 1:
                    new_row = state[0] + 1
                    new_col = state[1] - 1

            elif action == 9:
                propo = self.desc[state[0], 2 * state[1]] == b":" or self.desc[state[0] + 1, 2 * state[1]] == b":"
                if propo and state[0] >= 1 and state[1] >= 1:
                    new_row = state[0] - 1
                    new_col = state[1] - 1


        next_state = (new_row, new_col, new_pass_idx, dest_id)
        next_state = self.encode(next_state)
        self.current = next_state
        return (next_state, reward, done)


    def encode(self, state):
    # (5) 5, 5, 4
        i = state[0]
        i *= 5
        i += state[1]
        i *= 5
        i += state[2]
        i *= 4
        i += state[3]
        return i


    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return tuple(reversed(out))


    def reset(self):
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        pass_id = random.randint(0, 3)
        dest_id = random.randint(0, 3)
        while dest_id == pass_id:
            dest_id = random.randint(0, 3)
        state = self.encode((row, col, pass_id, dest_id))
        self.current = state
        return state


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.current)

        def ul(x): return "_" if x == " " else x
        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.dest[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        di, dj = self.dest[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


    # Simple transition function: remove 2 random walls
    def transition(self, walls_to_cut=None):
        if walls_to_cut is None:
            alternate = copy.deepcopy(self)
            walls = random.sample(self.walls, 1)
            for wall in walls:
                wall = tuple(wall)
                alternate.desc[wall[0]][wall[1]] = b":"
                alternate.walls.remove(wall)
            return alternate
            
        else:
            alternate = copy.deepcopy(self)
            for wall in walls_to_cut:
                walls = tuple(wall)
                alternate.walls.remove(wall)
                alternate.desc[wall[0]][wall[1]] = b":"
            return alternate


    def resettable_states(self):
        res = []
        for i in range(500):
            bd = self.decode(i)
            if 0 <= bd[0] <= 4 and 0 <= bd[1] <= 4 and 0 <= bd[2] <= 3 and 0 <= bd[3] <= 3 and bd[2] != bd[3]:
                res.append(i)

        return res


#map_to_numpy = np.asarray(map, dtype='c')
#env = TaxiEnv(map_to_numpy)
#env = env.transition([(1, 4)])
#env.reset()
#env.render()