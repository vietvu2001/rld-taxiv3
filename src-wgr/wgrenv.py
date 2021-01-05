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

        self.wind = [0, 1, 2, -1, 2, 1, 1, 0]

        # Destination: there are now three destinations
        self.dest = [(3, 5), (1, 6), (4, 7)]

        # Number of states
        self.num_states = self.width * self.length

        self.current = None

        # The fields for modifications
        self.special = []  # list of special cells that allow diagonal moves
        self.columns = []  # columns in which agent can move to any square


    def terminal(self, state):
        pos = (state[0], state[1])

        if pos == self.dest[state[2]]:
            return True

        return False    

    
    def reset(self):
        # The states we can reset to are only ones without column wind
        row = random.randint(0, self.max_row)
        col = random.randint(0, 1)

        # Choose a destination
        index = random.randint(0, 2)

        # Establish a state
        state = (row, col, index)

        self.current = copy.deepcopy(state)

        return state


    def step(self, action):
        assert self.current is not None
        
        # Default actions:
        # 0: move south
        # 1: move north
        # 2: move east
        # 3: move west

        i, j, goal_ind = self.current  # initialization
        row_opts = [i for i in range(self.width) if i != self.current[0]]
        next_state = self.current

        # Present reward
        reward = -1

        if action == 0:  # move south
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), j, goal_ind)

        elif action == 1:  # move north
            next_state = (max(i - 1 - self.wind[j], 0), j, goal_ind)

        elif action == 2:  # move east
            next_state = (max(i - self.wind[j], 0), min(j + 1, self.max_col), goal_ind)

        elif action == 3:  # move west
            next_state = (max(i - self.wind[j], 0), max(j - 1, 0), goal_ind)


        # Novel actions:
        # 4: move northeast
        # 5: move southeast
        # 6: move southwest
        # 7: move northwest
        
        elif action == 4 and self.current in self.special:  # move northeast
            next_state = (max(i - 1 - self.wind[j], 0), min(j + 1, self.max_col), goal_ind)

        elif action == 5 and self.current in self.special:  # move southeast
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), min(j + 1, self.max_col), goal_ind)

        elif action == 6 and self.current in self.special:  # move southwest
            next_state = (max(min(i + 1 - self.wind[j], self.max_row), 0), max(j - 1, 0), goal_ind)

        elif action == 7 and self.current in self.special:  # move northwest
            next_state = (min(i - 1 - self.wind[j], 0), max(j - 1, 0), goal_ind)


        # Column actions

        elif j in self.columns and action >= 8:
            assert action <= 13, "action larger than upper bound 13"
            next_state = (row_opts[action - 8], j, goal_ind)

        self.current = next_state
        done = self.terminal(next_state)

        return (next_state, reward, done) 

    
    def resettable_states(self):
        ls = []

        for i in range(self.width):
            for j in range(2):
                for index in range(3):
                    ls.append((i, j, index))

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
                if (self.current[0], self.current[1]) == position:
                    output = colored("X", "green")
                elif position == self.dest[self.current[2]]:
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

    
    def transition(self, columns_to_add):
        # Parameters
        # =========================================
        # columns_to_add: the columns in which we allow the agent to teleport to any square
        # =========================================

        ref = copy.deepcopy(self)

        for elem in columns_to_add:
            ref.columns.append(elem)

        return ref

'''
env = WindyGridworld()
env.reset()
print(env.current)

a = env.step(2)
print(env.current)
print(a)
env.render()

ls = env.resettable_states()
print(len(ls))'''