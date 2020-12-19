import os

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
from termcolor import colored  # pylint: disable=import-error

map = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

max_layer = 6

gamma = 1  # Discount factor for past rewards
max_steps_per_episode = 3000
eps = 1e-6

def utility(agent):
    # Mean of rewards coming from every resettable state
    # Parameters
    # ===================================================
    # agent: pre-trained q-learning agent
    # ===================================================
    rewards = []
    starts = agent.env.resettable_states()
    for point in starts:
        r = agent.eval(fixed=point, show=False)[1]
        rewards.append(r)

    return np.mean(rewards)


def make_env(env, mod_seq):
    # Make a new environment from reference original environment and a modification sequence
    # Parameters
    # ====================================================
    # env: original environment
    # mod_seq: modification sequence. As in list of 3-tuples (where first entry is indicator variable of special cell)
    # ====================================================
    ref_env = copy.deepcopy(env)
    locations = mod_seq
    for element in locations:
        if element[0] == 0:
            ref_env = ref_env.transition([(element[1], element[2])])
        else:
            ref_env.special.append((element[1], element[2]))
    
    return ref_env


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
        # There is a list of all possible modifications, called tree.modifications (stored in the MCTS tree)
        # This function updates the environment attributes of the node (remaining walls, special cells)

        # Parameters
        # ===========================================
        # self: the node
        # tree: the tree this node belongs to
        # parent_bool: whether this node is the root of the tree. boolean variable
        # ===========================================

        if not parent_bool: 
            return  # the root has no modification

        parent_env = copy.deepcopy(tree.nodes[self.parent].env)
        self.env = parent_env

        if tree.modifications[self.modification][0] == 0:  # wall, use transition function
            assert tuple(tree.modifications[self.modification][1 : 3]) in self.env.walls
            wall = tuple(tree.modifications[self.modification][1 : 3])
            self.env = self.env.transition([wall])

        elif tree.modifications[self.modification][0] == 1:  # cell, use append to special list
            assert tuple(tree.modifications[self.modification][1 : 3]) not in self.env.special
            cell = tuple(tree.modifications[self.modification][1 : 3])
            self.env.special.append(cell)


    def update_layer(self, tree):
        # This function finds out the layer of the node in the MCTS tree (after appending)

        if self.parent is None:  # or the root
            self.layer = 0

        else:  # it has a parent, so layer is the increment of parent's layer
            parent_layer = tree.nodes[self.parent].layer
            self.layer = parent_layer + 1


    def get_available_modifications(self, tree):
        # Get the available modifications immediately succeeding the node "self" in the tree
        # The idea here is that since every permutation of the modification sequence leads to the same utility of the agent
        # We should only accept only one configuration, here we are ordering the modification sequence by index
        # For example, (1, 15, 16, 19)
        ls = []

        # Maximum index from a node: if the node in on layer i, then it has only (max layer - i) more layers below it
        # Every layer means a modification (a path from that node), so the maximum index is bounded from above
        # For example, a node on layer 1 cannot have modification index 31, because it cannot increase as it goes down path
        # List is increasing so lower bound is the modification index of node + 1

        max_mod = len(tree.modifications) + self.layer - max_layer
        
        if self.parent is None:
            min_mod = 0
        
        else:
            min_mod = self.modification + 1

        for i in range(min_mod, max_mod + 1):
            ls.append(i)

        return ls


    def get_unused_modifications(self, tree):
        # A node has many possible modifications succeeding it on the tree (next layer following node)
        # This function finds out the modifications that are still unused (unvisited children)
        ls = self.get_available_modifications(tree)
        ls_ret = copy.deepcopy(ls)

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
        
        # List of all possible modifications: tree.modifications
        for wall in env.walls:
            self.modifications.append((0, wall[0], wall[1]))
        for row in range(env.width):
            for col in range(env.length):
                self.modifications.append((1, row, col))
        
        self.num_nodes = 0
        self.root = None
        self.threshold = 9


    def scale(self, x):
        # Scale the utility of the agent for backpropagation
        return max(0, x - self.threshold)


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


    def best_child(self, node_index, const, const_2=1, expanded=True):  # return index of best child according to ucb heuristic
        assert self.num_nodes > node_index
        ls = self.nodes[node_index].get_unused_modifications(self)
        if expanded:
            assert len(ls) == 0

        opt = float("-inf")
        child = None
        for c in self.nodes[node_index].visited_children:
            scaled_reward = self.nodes[c].sum_reward / self.nodes[c].count
            exploration_term = const * math.sqrt(2 * math.log(self.nodes[node_index].count) / self.nodes[c].count)
            extra = 0
            if len(self.nodes[c].simulation_history) != 0:
                extra = const_2 * math.sqrt(np.var(self.nodes[c].simulation_history) + 1 / self.nodes[c].count)

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
        # We know that tree.nodes[start] is a leaf, so there is no used modifications at start yet.
        ls = []
        for i in range(self.nodes[start].modification + 1, len(self.modifications)):
            ls.append(i)

        a = random.sample(ls, k=mods_left)
        a = sorted(a)
        for element in a:
            mod = self.modifications[element]
            if mod[0] == 0:
                simulate_env = simulate_env.transition([(mod[1], mod[2])])
            elif mod[0] == 1:
                simulate_env.special.append((mod[1], mod[2]))
        
        # Training
        agent = QAgent(simulate_env)
        agent.qlearn(600, render=False)
        reward = utility(agent)

        if reward > self.threshold:
            print(colored(a, "red"))
            for element in a:
                start = self.add_node(element, start).index
            
            return [self.scale(reward), start]

        return self.scale(reward)

    
    def tree_policy(self, node_index):
        iter_index = node_index
        while not self.nodes[iter_index].terminal():
            if not self.nodes[iter_index].fully_expanded(self):
                return self.expand(iter_index)
            
            else:
                iter_index = self.best_child(iter_index, 0.8, 0.8)
        
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
            print(colored("Iteration {} begins!".format(i), "red"))
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
        while self.nodes[start].layer < max_layer:
            if len(self.nodes[start].visited_children) != 0:
                start = self.best_child(start, 0, 0, expanded=False)
                mod_index = self.nodes[start].modification
                walk.append(self.modifications[mod_index])
        
        if len(walk) < max_layer:
            print("MCTS insufficient to get {} modifications!".format(max_layer))
            return (walk, None)

        else:
            modified = make_env(self.env, walk)
            agent = QAgent(modified)
            agent.qlearn(600, render=False)
            rews = utility(agent)
            return (walk, rews)


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
tree.ucb_search(iterations=2)

# Store data
'''r_dir = os.path.abspath(os.pardir)
data_dir = os.path.join(r_dir, "data")
csv_dir = os.path.join(data_dir, "tree_{}.csv".format(max_layer))
txt_dir = os.path.join(data_dir, "mcts_result_{}.txt".format(max_layer))

with open(csv_dir, "w", newline='') as file:
    fieldnames = list(tree.info(0).keys())
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for node in tree.nodes:
        writer.writerow(tree.info(node.index))

a = tree.greedy()

with open(txt_dir, "w") as file:
    file.write("Modifications: ")
    file.write(str(a[0]))
    file.write("\n")
    file.write("Utility: ")
    if a[1] is not None:
        file.write(str(a[1]))
    else:
        file.write("Utility not available")'''

print(tree.root.layer)