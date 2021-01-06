import numpy as np  # pylint: disable=import-error
import random
import gym  # pylint: disable=import-error
import time
from collections import deque
import copy
import sys
from termcolor import colored  # pylint: disable=import-error


class WindyGridworld():
    def __init__(self):
        self.width = 7
        self.length = 8

        self.max_row = self.width - 1
        self.max_col = self.length - 1

        self.wind = [0, 0, 1, 1, 1, 0, 0, 0]

        # Destination: there are now three destinations
        self.dest = [(1, 6), (3, 5), (4, 7)]

        # Number of states
        self.num_states = self.width * self.length

        self.current = None

        # The fields for modifications
        self.special = []  # list of special cells that allow diagonal moves
        self.jump_cells = []  # list of cells that allow the agent to jump

    # Notes:
    # A state is a four-tuple (x, y, first_addr, second_addr) where:
    # (x, y): current position of the agent
    # first_addr: the index of the first goal the agent has to reach
    # second_addr: the index of the second goal the agent has to reach
    # (x, y) satisfies 0 <= x <= 6 and 0 <= y <= 7
    # Normally, first_addr and second_addr would lie in the interval [0, 2]. But if first_addr is 3, that means the first goal is already reached.


    def terminal(self, state):
        if state[2] == 3 and (state[0], state[1]) == self.dest[state[3]]:
            return True

        return False

    
    def reset(self):
        # The states we can reset to are only ones without column wind
        row = random.randint(0, self.max_row)
        col = random.randint(0, 1)

        # Choose two goals with order
        goals = random.sample([0, 1, 2], k=2)

        # Establish a state
        state = (row, col, goals[0], goals[1])

        self.current = copy.deepcopy(state)

        return state


    def step(self, action):
        assert self.current is not None
        assert len(self.current) == 4
        assert 0 <= action <= 11
        
        # Default actions:
        # 0: move south
        # 1: move north
        # 2: move east
        # 3: move west

        i, j, f_addr, s_addr = self.current  # initialization
        next_state = self.current

        # Present reward
        reward = -1

        # Preprocess conditionals
        special_cell = (i, j) in self.special
        jump_cell = (i, j) in self.jump_cells

        if action == 0:  # move south
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), j, f_addr, s_addr)

        elif action == 1:  # move north
            next_state = (max(i - 1 - self.wind[j], 0), j, f_addr, s_addr)

        elif action == 2:  # move east
            next_state = (max(i - self.wind[j], 0), min(j + 1, self.max_col), f_addr, s_addr)

        elif action == 3:  # move west
            next_state = (max(i - self.wind[j], 0), max(j - 1, 0), f_addr, s_addr)


        # Novel actions:
        # 4: move northeast
        # 5: move southeast
        # 6: move southwest
        # 7: move northwest

        
        elif action == 4 and special_cell:  # move northeast
            next_state = (max(i - 1 - self.wind[j], 0), min(j + 1, self.max_col), f_addr, s_addr)

        elif action == 5 and special_cell:  # move southeast
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), min(j + 1, self.max_col), f_addr, s_addr)

        elif action == 6 and special_cell:  # move southwest
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), max(j - 1, 0), f_addr, s_addr)

        elif action == 7 and special_cell:  # move northwest
            next_state = (min(i - 1 - self.wind[j], 0), max(j - 1, 0), f_addr, s_addr)


        # Novel actions (continued)
        # 8: jump south 2 steps
        # 9: jump north 2 steps
        # 10: jump east 2 steps
        # 11: jump west 2 steps

        elif action == 8 and jump_cell:
            next_state = (max(min(i + 2 - self.wind[j], self.max_row), 0), j, f_addr, s_addr)

        elif action == 9 and jump_cell:
            next_state = (max(i - 1 - self.wind[j], 0), j, f_addr, s_addr)

        elif action == 10 and jump_cell:
            next_state = (max(i - self.wind[j], 0), min(j + 2, self.max_col), f_addr, s_addr)

        elif action == 11 and jump_cell:
            next_state = (max(i - self.wind[j], 0), max(j - 2, 0), f_addr, s_addr)

        if f_addr != 3:
            if (next_state[0], next_state[1]) == self.dest[f_addr]:
                self.index_for_render = f_addr
                next_state = list(next_state)
                next_state[2] = 3
                next_state = tuple(next_state)

        self.current = next_state
        done = self.terminal(next_state)

        if done:
            reward = 20

        return (next_state, reward, done) 

    
    def resettable_states(self):
        ls = []

        for i in range(self.width):
            for j in range(2):
                for ind_1 in range(3):
                    for ind_2 in range(3):
                        if ind_1 != ind_2:
                            ls.append((i, j, ind_1, ind_2))

        return ls


    def render(self, mode='human', close=False):
        ''' Renders the environment. Code borrowed and then modified 
            from
            https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py'''

        outfile = sys.stdout
        shape = (self.width, self.length)

        outboard = ""
        assert self.current is not None

        for y in range(-1, self.width + 1):
            outline = ""
            for x in range(-1, self.length + 1):
                position = (y, x)
                if (self.current[0], self.current[1]) == position:
                    output = colored("X", "green")
                elif self.current[2] != 3 and position == self.dest[self.current[2]]:
                    output = colored("G", "magenta")

                elif position == self.dest[self.current[3]]:
                    output = colored("G", "red")
                elif position in self.dest:
                    output = "G"
                elif x in {-1, self.length} or y in {-1, self.width}:
                    output = "#"
                else:
                    output = " "

                if position[1] == shape[1]:
                    output += '\n'

                outline += output
            outboard += outline
        outboard += '\n'
        outfile.write(outboard)

'''
env = WindyGridworld()
env.reset()
env.current = (1, 0, 2, 1)
env.render()

actions = [2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 1, 3, 3]

s = None

for a in actions:
    s = env.step(a)

print(env.current)
print(s)

env.render()'''