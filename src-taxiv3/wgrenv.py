import numpy as np  # pylint: disable=import-error
import random
import gym  # pylint: disable=import-error
import time
from collections import deque
import copy
import sys


class WindyGridworld():
    def __init__(self):
        self.width = 7
        self.length = 10

        self.max_row = self.width - 1
        self.max_col = self.length - 1

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # Destination: there is only one destination
        self.dest = [(3, 7)]

        # Number of states
        self.num_states = self.width * self.length

        self.current = None

        # The fields for modifications
        self.special = []  # list of special cells that allow diagonal moves
        self.columns = []  # columns in which agent can move to any square

        self.start_state = None


    def terminal(self, state):
        if state in self.dest:
            return True

        return False    

    
    def reset(self):
        # The states we can reset to are only ones without column wind
        row = random.randint(0, self.max_row)
        col = random.randint(0, 2)

        # Establish a state
        state = (row, col)

        self.current = copy.deepcopy(state)
        self.start_state = copy.deepcopy(state)


        return state


    def step(self, action):
        assert self.current is not None
        
        # Default actions:
        # 0: move south
        # 1: move north
        # 2: move east
        # 3: move west

        i, j = self.current  # initialization
        row_opts = [i for i in range(self.width) if i != self.current[0]]

        if action == 0:  # move south
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), j)

        elif action == 1:  # move north
            next_state = (max(i - 1 - self.wind[j], 0), j)

        elif action == 2:  # move east
            next_state = (max(i - self.wind[j], 0), min(j + 1, self.max_col))

        elif action == 3:  # move west
            next_state = (max(i - self.wind[j], 0), max(j - 1, 0))


        # Novel actions:
        # 4: move northeast
        # 5: move southeast
        # 6: move southwest
        # 7: move northwest
        

        elif action == 4 and self.current in self.special:  # move northeast
            next_state = (max(i - 1 - self.wind[j], 0), min(j + 1, self.max_col))

        elif action == 5 and self.current in self.special:  # move southeast
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), min(j + 1, self.max_col))

        elif action == 6 and self.current in self.special:  # move southwest
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), max(j - 1, 0))

        elif action == 7 and self.current in self.special:  # move northwest
            next_state = (min(i - 1 - self.wind[j], 0), max(j - 1, 0))


        # Column actions

        elif j in self.columns:
            next_state = (row_opts[action - 8], j)

        self.current = next_state
        return next_state 

    
    def resettable_states(self):
        ls = []

        for i in range(self.width):
            for j in range(3):
                ls.append((i, j))

        return ls

    def render(self, mode='human', close=False):
        ''' Renders the environment. Code borrowed and then modified 
            from
            https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py'''

        outfile = sys.stdout
        shape = (self.width, self.length)

        outboard = ""
        for y in range(-1, self.width + 1):
            outline = ""
            for x in range(-1, self.length + 1):
                position = (y, x)
                if self.current == position:
                    output = "X"
                elif position in self.dest:
                    output = "G"
                elif position == self.start_state:
                    output = "S"
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


env = WindyGridworld()
env.reset()
env.step(2)
env.step(2)
env.step(2)
print("Current state: {}".format(env.current))
print("Starting state: {}".format(env.start_state))
env.render()