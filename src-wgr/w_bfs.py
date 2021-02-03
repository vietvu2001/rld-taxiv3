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
    locations = copy.deepcopy(mod_seq)

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

        if tree.modifications[self.modification][0] == 0:  # jump cell, use append to jump cells list
            assert tuple(tree.modifications[self.modification][1 : 3]) not in self.env.jump_cells
            jump = tuple(tree.modifications[self.modification][1 : 3])
            self.env.jump_cells.append(jump)

        elif tree.modifications[self.modification][0] == 1:  # special cell, use append to special list
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

        # Maximum index from a node: if the node is on layer i, then it has only (max layer - i) more layers below it
        # Every layer means a modification (a path from that node), so the maximum index is bounded from above
        # For example, a node on layer 1 cannot have modification index 31, because it cannot increase as it goes down path
        # List is increasing so lower bound is the modification index of node + 1

        max_mod = len(tree.modifications) + self.layer - tree.max_layer
        
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
        
        # List of all possible modifications: tree.modifications
        for row in range(env.width):
            for col in range(env.length):
                self.modifications.append((0, row, col))

        for row in range(env.width):
            for col in range(env.length):
                self.modifications.append((1, row, col))
        
        self.num_nodes = 0
        self.root = None
        self.threshold = 10.12

        self.max_layer = max_layer

        # Storing tree's max reward and corresponding environment
        self.max_reward = float("-inf")
        self.opt_env = None


    def scale(self, x):
        # Scale the utility of the agent for backpropagation
        return max(0, x - self.threshold)


    def initialize(self):
        # Initialize the tree with the root
        # Assert that the tree is null (no root beforehand)
        assert self.root == None

        # Create root
        root = Node(None, self.num_nodes, None, self.env)

        # Add root to the list of nodes of the tree and increase the number of nodes 
        self.nodes.append(root)
        self.num_nodes += 1

        # Assign the root to the root field of the tree
        self.root = root

        # Update the environment configuration of the root
        self.root.update_jumps_special(self, parent_bool=False)

        # Update the layer of the root (the tree's version)
        self.root.update_layer(self)


    def add_node(self, mod_index, parent_index):
        # This function adds a node with a modification index and a parent index into the MCTS tree.
        assert parent_index < self.num_nodes
        assert mod_index in self.nodes[parent_index].get_unused_modifications(self)

        # Create a node from the modification index (in tree.modifications) and the parent index on the tree
        node = Node(mod_index, self.num_nodes, parent_index, self.env)

        # Append the node to the list of nodes of the tree and increase the number of nodes
        self.nodes.append(node)
        self.num_nodes += 1

        # Changing the boolean leaf status of the parent_index to False. 
        self.nodes[parent_index].leaf = False

        # Update the layer of the newly added node
        self.nodes[node.index].update_layer(self)

        # Update the environment configuration of the newly added node
        self.nodes[node.index].update_jumps_special(self, parent_bool=True)

        # Mark the newly added node as a visited child, with respect to its parent node
        self.nodes[parent_index].visited_children.append(node.index)

        # Return the node as output
        return self.nodes[node.index]


    def default_policy(self, node_index):
        start = node_index
        simulate_env = copy.deepcopy(self.nodes[start].env)
        num_modifications_applied = len(simulate_env.jump_cells) + len(simulate_env.special) - len(self.env.special) - len(self.env.jump_cells)
        mods_left = self.max_layer - num_modifications_applied
        
        # Choose from unused modifications, from start node
        # We know that tree.nodes[start] is a leaf, so there is no used modifications at start yet.
        ls = []
        
        if node_index != 0:
            for i in range(self.nodes[start].modification + 1, len(self.modifications)):
                ls.append(i)

        else:
            ls = [i for i in range(len(self.modifications))]

        a = random.sample(ls, k=mods_left)
        a = sorted(a)
        for element in a:
            mod = self.modifications[element]

            if mod[0] == 0:
                simulate_env.jump_cells.append((mod[1], mod[2]))
            elif mod[0] == 1:
                simulate_env.special.append((mod[1], mod[2]))
        
        # Training
        agent = w_QAgent(simulate_env)
        agent.qlearn(3000, show=False)
        reward = utility(agent)

        if reward > self.max_reward:
            self.max_reward = reward
            self.opt_env = copy.deepcopy(simulate_env)

        return reward


    def BFS(self, num_iters):
        queue = []

        # Add root to queue
        queue.append(0)
        count = 0

        while count < num_iters and len(queue) != 0:
            print(colored("Iteration {} begins!".format(count), "red"))
            index = queue.pop(0)  # pop first element out of the queue

            # Run default policy
            r = self.default_policy(index)

            print("Reward: {}".format(r))

            # Add all its children to the queue
            if not self.nodes[index].terminal(self):
                ls = self.nodes[index].get_unused_modifications(self)  # list of modifcation indices
                for mod_index in ls:
                    self.add_node(mod_index, index)
            
                    # Add newly created node to queue
                    queue.append(len(self.nodes) - 1)  # newly created node must have highest index in the node list of tree
            
            count += 1

            # Print statistics
            print(colored("Number of nodes seen so far: {}".format(len(self.nodes)), "green"))
            print(colored("Maximum reward seen so far: {}".format(self.max_reward), "green"))
    

    def best_observed_choice(self):
        vector = []
        for jump in self.opt_env.jump_cells:
            if jump not in self.env.jump_cells:
                tup = (0, jump[0], jump[1])
                vector.append(tup)

        for cell in self.opt_env.special:
            if cell not in self.env.special:
                tup = (1, cell[0], cell[1])
                vector.append(tup)

        # Training to prevent errors arising from connected training
        agent = w_QAgent(self.opt_env)
        agent.qlearn(3500)
        rews = utility(agent)

        x = max(rews, self.max_reward)

        return (vector, x)


if __name__ == "__main__":
    max_layer = int(sys.argv[1])
    num_iters = int(sys.argv[2])

    env = WindyGridworld()
    
    tree = Tree(env, max_layer)
    tree.threshold = float(sys.argv[3])
    tree.initialize()

    tree.BFS(num_iters)

    # Store data
    r_dir = os.path.abspath(os.pardir)
    data_dir = os.path.join(r_dir, "data-wgr")
    txt_dir = os.path.join(data_dir, "bfs_result_{}.txt".format(tree.max_layer))

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