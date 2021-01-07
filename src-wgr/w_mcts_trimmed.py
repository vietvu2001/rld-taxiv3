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

from wgrenv import WindyGridworld
from w_qlearn import w_QAgent
from termcolor import colored  # pylint: disable=import-error
from w_heuristic import cell_frequency

max_layer = 1


def utility(agent):
    rewards = []
    starts = agent.env.resettable_states()
    for point in starts:
        r = agent.eval(fixed=point, show=False)[1]
        rewards.append(r)

    return np.mean(rewards)


def make_env(env, mod_seq):
    ref_env = copy.deepcopy(env)
    locations = mod_seq
    for element in locations:
        if element[0] == 0:
            ref_env.jump_cells.append((element[1], element[2]))
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


    def update_jumps_special(self, tree, parent_bool):
        if not parent_bool: 
            return

        parent_env = copy.deepcopy(tree.nodes[self.parent].env)
        self.env = parent_env
        if tree.modifications[self.modification][0] == 0:  # wall
            assert tuple(tree.modifications[self.modification][1 : 3]) not in self.env.jump_cells
            jump = tuple(tree.modifications[self.modification][1 : 3])
            self.env.jump_cells.append(jump)

        elif tree.modifications[self.modification][0] == 1:  # cell
            assert tuple(tree.modifications[self.modification][1 : 3]) not in self.env.special
            cell = tuple(tree.modifications[self.modification][1 : 3])
            self.env.special.append(cell)


    def update_layer(self, tree):
        if self.parent is None:
            self.layer = 0
        else:
            parent_layer = tree.nodes[self.parent].layer
            self.layer = parent_layer + 1


    def get_available_modifications(self, tree):
        ls = []
        max_mod = len(tree.modifications) + self.layer - tree.max_layer
        if self.parent is None:
            min_mod = 0
        
        else:
            min_mod = self.modification + 1

        for i in range(min_mod, max_mod + 1):
            ls.append(i)

        return ls


    def get_unused_modifications(self, tree):
        ls = self.get_available_modifications(tree)
        ls_ret = ls
        for index in self.visited_children:
            mod_index = tree.nodes[index].modification
            ls_ret.remove(mod_index)
        
        return ls_ret


    def terminal(self, tree):
        return self.layer == tree.max_layer

    
    def fully_expanded(self, tree):
        ls = self.get_unused_modifications(tree)
        return len(ls) == 0


class Tree():
    def __init__(self, env, max_layer):
        self.env = env
        self.modifications = []
        self.counter = 0
        self.nodes = []

        agent = w_QAgent(env)
        agent.qlearn(2000 + 200 * (max_layer - 1), render=False)
        cell_dict = cell_frequency(agent)
        
        for element in cell_dict[0 : 15]:
            self.modifications.append((0, element[0][0], element[0][1]))

        for element in cell_dict[0 : 15]:
            self.modifications.append((1, element[0][0], element[0][1]))
        
        self.num_nodes = 0
        self.root = None
        self.max_layer = max_layer
        self.threshold = 10.12

        # Storing best reward and corresponding environment
        self.max_reward = float("-inf")
        self.opt_env = None


    def scale(self, x):
        return max(0, x - self.threshold)


    def initialize(self):
        assert self.root == None
        root = Node(None, self.num_nodes, None, self.env)
        self.nodes.append(root)
        self.num_nodes += 1
        self.root = root
        self.root.update_jumps_special(self, parent_bool=False)
        self.root.update_layer(self)


    def add_node(self, mod_index, parent_index):
        assert parent_index < self.num_nodes
        assert mod_index in self.nodes[parent_index].get_unused_modifications(self)
        node = Node(mod_index, self.num_nodes, parent_index, self.env)
        self.nodes.append(node)
        self.num_nodes += 1
        self.nodes[parent_index].leaf = False
        self.nodes[node.index].update_layer(self)
        self.nodes[node.index].update_jumps_special(self, parent_bool=True)
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

        chosen_mod = self.modifications[self.nodes[child].modification]
        print(colored("Chosen child's modification: {}".format(chosen_mod), "red"))
        
        return child


    def default_policy(self, node_index):
        start = node_index
        simulate_env = copy.deepcopy(self.nodes[start].env)
        num_modifications_applied = len(simulate_env.jump_cells) + len(simulate_env.special) - len(self.env.special) - len(self.env.jump_cells)
        mods_left = self.max_layer - num_modifications_applied
        
        # Choose from unused modifications, from start node
        # We know that tree.nodes[start] is a leaf, so there is no used modifications at start yet.
        ls = []
        for i in range(self.nodes[start].modification + 1, len(self.modifications)):
            ls.append(i)

        try:
            a = random.sample(ls, k=mods_left)

        except:
            print(ls)
            print(num_modifications_applied)
            raise ValueError

        a = sorted(a)
        for element in a:
            mod = self.modifications[element]
            if mod[0] == 0:
                simulate_env.jump_cells((mod[1], mod[2]))
            elif mod[0] == 1:
                simulate_env.special.append((mod[1], mod[2]))
        
        # Training
        agent = w_QAgent(simulate_env)
        agent.qlearn(2000 + 200 * (self.max_layer - 1), render=False)
        reward = utility(agent)

        if reward > self.threshold:
            print(colored(a, "red"))
            print(colored(reward, "red"))
            for element in a:
                start = self.add_node(element, start).index

            # Update tree's max reward if possible
            if reward > self.max_reward:
                self.max_reward = reward
                self.opt_env = simulate_env
            
            return [self.scale(reward), start]

        return self.scale(reward)

    
    def tree_policy(self, node_index, c1, c2):
        iter_index = node_index
        while not self.nodes[iter_index].terminal(self):
            if not self.nodes[iter_index].fully_expanded(self):
                return self.expand(iter_index)
            
            else:
                iter_index = self.best_child(iter_index, c1, c2)
        
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
        c1 = 1
        c2 = 1

        for i in range(iterations):
            print(colored("Iteration {} begins!".format(i), "red"))
            leaf_index = self.tree_policy(root_index, c1, c2)
            a = self.default_policy(leaf_index)
            if isinstance(a, list):
                leaf_index = a[1]
                reward = a[0]
            
            else:
                reward = a

            self.backup(leaf_index, reward)
            print(colored("Number of nodes so far: {}".format(len(self.nodes)), "green"))
            print(colored("Maximum reward seen so far: {}".format(self.max_reward), "green"))
            print("Iteration {} ends!".format(i))
            print()


    def greedy(self):
        walk = []
        start = 0
        while self.nodes[start].layer < self.max_layer:
            if len(self.nodes[start].visited_children) != 0:
                start = self.best_child(start, 0, 0, expanded=False)
                mod_index = self.nodes[start].modification
                walk.append(self.modifications[mod_index])
        
        if len(walk) < self.max_layer:
            print("MCTS insufficient to get {} modifications".format(self.max_layer))
            return (walk, None)

        else:
            modified = make_env(self.env, walk)
            agent = w_QAgent(modified)
            agent.qlearn(2000 + 200 * (self.max_layer - 1), render=False)
            rews = utility(agent)
            return (walk, rews)

    
    def best_observed_choice(self):
        vector = []
        for jump in self.opt_env.jump_cells:
            if jump not in self.opt_env.walls:
                tup = (0, jump[0], jump[1])
                vector.append(tup)

        for cell in self.opt_env.special:
            if cell not in self.env.special:
                tup = (1, cell[0], cell[1])
                vector.append(tup)

        # Training to prevent errors arising from connected training
        agent = w_QAgent(self.opt_env)
        agent.qlearn(2000 + 200 * (self.max_layer - 1))
        rews = utility(agent)

        return (vector, rews)


    def info(self, node_index):
        dict_return = {}
        for key in vars(self.nodes[node_index]):
            if key != "simulation_history":
                if key != "env":
                    dict_return[key] = vars(self.nodes[node_index])[key]
                else:
                    dict_return["jump_cells"] = self.nodes[node_index].env.jump_cells
                    dict_return["special_cells"] = self.nodes[node_index].env.special
        
        return dict_return


if __name__ == "__main__":
    num_iters = 120
    env = WindyGridworld()
    tree = Tree(env, max_layer)
    tree.initialize()
    tree.ucb_search(iterations=num_iters)
    r_dir = os.path.abspath(os.pardir)
    data_dir = os.path.join(r_dir, "data-wgr")
    txt_dir = os.path.join(data_dir, "mcts_trimmed_result_{}.txt".format(tree.max_layer))

    a = tree.best_observed_choice()

    with open(txt_dir, "w") as file:
        file.write("Modifications: ")
        file.write(str(a[0]))
        file.write("\n")
        file.write("Utility: ")
        if a[1] is not None:
            file.write(str(a[1]))
        else:
            file.write("Utility not available")

        file.write("\n")
        file.write("Number of iterations: {}".format(num_iters))
        file.write("\n")
        file.write("Threshold: {}".format(tree.threshold))