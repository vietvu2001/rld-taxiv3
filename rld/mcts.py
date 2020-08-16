import numpy as np  # pylint: disable=import-error
import random
from gym import utils  # pylint: disable=import-error
from io import StringIO
import sys
from contextlib import closing
import copy
import math
import csv

import gym  # pylint: disable=import-error
import time
import random
from collections import deque

from taxienv import TaxiEnv
from qlearn import QAgent

map = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

max_layer = 4

gamma = 1  # Discount factor for past rewards
max_steps_per_episode = 3000
eps = 1e-6

class Node():
    def __init__(self, mod_index, index, parent_index, env):
        if mod_index is not None:
            self.modification = mod_index
        else:
            self.modification = None
        if parent_index is not None:
            self.parent = parent_index
        else:
            self.parent = None

        self.index = index
        self.visited_children = []
        self.env = copy.deepcopy(env)
        self.leaf = True
        self.sum_reward = 0
        self.count = 0
        self.layer = -1
        self.simulation_history = []


    def update_walls_special(self, tree, parent_bool):
        if not parent_bool: 
            return

        parent_env = copy.deepcopy(tree.nodes[self.parent].env)
        self.env = parent_env
        if tree.modifications[self.modification][0] == "w":  # wall
            assert tree.modifications[self.modification][1] in self.env.walls
            #self.env.walls.remove(tree.modifications[self.modification][1])
            self.env = self.env.transition([tree.modifications[self.modification][1]])
        elif tree.modifications[self.modification][0] == "c":  # cell
            assert tree.modifications[self.modification][1] not in self.env.special
            self.env.special.append(tree.modifications[self.modification][1])


    def update_layer(self, tree):
        if self.parent is None:
            self.layer = 0
        else:
            parent_layer = tree.nodes[self.parent].layer
            self.layer = parent_layer + 1


    def get_available_modifications(self, tree):
        ls = []
        for wall in self.env.walls:
            mod_index = tree.modifications.index(("w", wall))
            ls.append(mod_index)
        
        for row in range(self.env.width):
            for col in range(self.env.length):
                if (row, col) not in self.env.special:
                    mod_index = tree.modifications.index(("c", (row, col)))
                    ls.append(mod_index)
        return ls


    def get_unused_modifications(self, tree):
        ls = self.get_available_modifications(tree)
        ls_ret = ls
        for index in self.visited_children:
            mod_index = tree.nodes[index].modification
            ls_ret.remove(mod_index)
        
        return ls_ret


    def terminal(self):
        return self.layer == max_layer

    
    def fully_expanded(self, tree):
        ls = self.get_unused_modifications(tree)
        return len(ls) == 0


class Tree():
    def __init__(self, env):
        self.env = env
        self.modifications = []
        self.counter = 0
        self.nodes = []
        for wall in env.walls:
            self.modifications.append(('w', wall))
        for row in range(env.width):
            for col in range(env.length):
                self.modifications.append(('c', (row, col)))
        
        self.num_nodes = 0
        self.root = None


    def scale(self, x):
        if x < 9:
            return 0
        elif x >= 9 and x < 10:
            return x - 9
        else:
            return 1


    def initialize(self):
        assert self.root == None
        root = Node(None, self.num_nodes, None, self.env)
        self.nodes.append(root)
        self.num_nodes += 1
        self.root = root
        self.root.update_walls_special(self, parent_bool=False)
        self.root.update_layer(self)


    def add_node(self, mod_index, parent_index):
        assert parent_index < self.num_nodes
        assert mod_index in self.nodes[parent_index].get_unused_modifications(self)
        node = Node(mod_index, self.num_nodes, parent_index, self.env)
        self.nodes.append(node)
        self.num_nodes += 1
        self.nodes[parent_index].leaf = False
        self.nodes[node.index].update_layer(self)
        self.nodes[node.index].update_walls_special(self, parent_bool=True)
        self.nodes[parent_index].visited_children.append(node.index)
        return self.nodes[node.index]


    def expand(self, node_index):  # return index of an expanded node
        assert self.num_nodes > node_index
        ls = self.nodes[node_index].get_unused_modifications(self)
        assert len(ls) > 0  # still have an unvisited child
        mod_index = random.choice(ls)
        node = self.add_node(mod_index, node_index)
        return node.index


    def best_child(self, node_index, const):  # return index of best child according to ucb heuristic
        assert self.num_nodes > node_index
        ls = self.nodes[node_index].get_unused_modifications(self)
        assert len(ls) == 0
        opt = float("-inf")
        child = None
        for c in self.nodes[node_index].visited_children:
            scaled_reward = self.scale(self.nodes[c].sum_reward / self.nodes[c].count)
            exploration_term = const * math.sqrt(2 * math.log(self.nodes[node_index].count) / self.nodes[c].count)
            extra = 0
            if len(self.nodes[c].simulation_history) != 0:
                extra = math.sqrt(np.var(self.nodes[c].simulation_history) + 1 / self.nodes[c].count)

            result = scaled_reward + exploration_term + extra  # Schadd SP-MCTS added term
            if result > opt:
                opt = result
                child = c
        
        return child


    def default_policy(self, node_index):
        start = node_index
        simulate_env = copy.deepcopy(self.nodes[start].env)
        num_modifications_applied = len(self.env.walls) - len(simulate_env.walls) + len(simulate_env.special)
        mods_left = max_layer - num_modifications_applied
        
        # Choose from unused modifications, from start node
        ls = self.nodes[start].get_unused_modifications(self)
        a = random.sample(ls, k=mods_left)
        for element in a:
            mod = self.modifications[element]
            if mod[0] == "w":
                simulate_env = simulate_env.transition([mod[1]])
            elif mod[0] == "c":
                simulate_env.special.append(mod[1])
        
        # Training
        total = 0
        agent = QAgent(simulate_env)
        agent.qlearn(600, show=True)
        for _ in range(500):
            result = agent.eval(show=True)[1]
            total += result

        reward = total / 500

        if reward > 9.9:
            print("Good choice!")
            for element in a:
                start = self.add_node(element, start).index
            
            return [reward, start]

        return reward

    
    def tree_policy(self, node_index):
        iter_index = node_index
        while not self.nodes[iter_index].terminal():
            if not self.nodes[iter_index].fully_expanded(self):
                return self.expand(iter_index)
            
            else:
                iter_index = self.best_child(iter_index, 1)
        
        return iter_index


    def backup(self, node_index, reward):
        iter_index = node_index
        while iter_index is not None:
            self.nodes[iter_index].sum_reward += reward
            self.nodes[iter_index].simulation_history.append(reward)
            self.nodes[iter_index].count += 1
            iter_index = self.nodes[iter_index].parent
    

    def ucb_search(self, iterations):
        root_index = self.nodes[0].index
        for i in range(iterations):
            print("Iteration {} begins!".format(i))
            leaf_index = self.tree_policy(root_index)
            a = self.default_policy(leaf_index)
            if isinstance(a, list):
                leaf_index = a[1]
                reward = a[0]
            
            else:
                reward = a

            self.backup(leaf_index, reward)
            print("Iteration {} ends!".format(i))
            print()


    def greedy(self):
        walk = []
        start = 0
        while self.nodes[start].fully_expanded(self):
            if len(self.nodes[start].get_unused_modifications(self)) == 0:
                start = self.best_child(start, 0)
                walk.append(self.nodes[start].modification)
        
        if len(walk) < max_layer:
            opt = float("-inf")
            child = None
            for c in self.nodes[start].visited_children:
                value = self.scale(self.nodes[c].sum_reward / self.nodes[c].count)
                if value > opt:
                    opt = value
                    child = c

            walk.append(self.nodes[child].modification)
        
        return walk


    def info(self, node_index):
        dict_return = {}
        for key in vars(self.nodes[node_index]):
            if key != "simulation_history":
                if key != "env":
                    dict_return[key] = vars(self.nodes[node_index])[key]
                else:
                    dict_return["walls"] = self.nodes[node_index].env.walls
                    dict_return["special_cells"] = self.nodes[node_index].env.special
        
        return dict_return


map_to_numpy = np.asarray(map, dtype='c')
env = TaxiEnv(map_to_numpy)
tree = Tree(env)
tree.initialize()
tree.ucb_search(iterations=1500)
with open("tree_2.csv", "w", newline='') as file:
    fieldnames = list(tree.info(0).keys())
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for node in tree.nodes:
        writer.writerow(tree.info(node.index))

a = tree.greedy()

with open("modifications.txt", "w") as file:
    for mod_index in a:
        file.write(str(tree.modifications[mod_index]))
        file.write("\n")

#print(tree.modifications)