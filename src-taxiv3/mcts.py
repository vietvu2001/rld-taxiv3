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

max_layer = 3

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
        for wall in env.walls:
            self.modifications.append((0, wall[0], wall[1]))
        for row in range(env.width):
            for col in range(env.length):
                self.modifications.append((1, row, col))
        
        self.num_nodes = 0
        self.root = None
        self.threshold = 8.5

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
        self.root.update_walls_special(self, parent_bool=False)

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
        self.nodes[node.index].update_walls_special(self, parent_bool=True)

        # Mark the newly added node as a visited child, with respect to its parent node
        self.nodes[parent_index].visited_children.append(node.index)

        # Return the node as output
        return self.nodes[node.index]


    def expand(self, node_index):  # return index of an expanded node
        # Get the list of unused modifications on the node (given by the tree and the node index)
        ls = self.nodes[node_index].get_unused_modifications(self)

        # Assert that the node still has unvisited children
        assert len(ls) > 0

        # Choose a random modification index from this list
        mod_index = random.choice(ls)

        # Add the node of this modification index into the tree
        node = self.add_node(mod_index, node_index)

        # Return the index of this newly created node as output
        return node.index


    def best_child(self, node_index, const, const_2=1, expanded=True):
        # Find the best child of a node based on the UCB heuristic
        # If node is not fully expanded, use the heuristic only on visited children
        assert self.num_nodes > node_index
        ls = self.nodes[node_index].get_unused_modifications(self)

        # If expanded boolean is True, the length of unused modifications list must be 0
        if expanded:
            assert len(ls) == 0

        # Find the best child with largest UCB value
        opt = float("-inf")
        child = None
        for c in self.nodes[node_index].visited_children:
            # Calculate the term for child c
            
            scaled_reward = self.nodes[c].sum_reward / self.nodes[c].count
            exploration_term = const * math.sqrt(2 * math.log(self.nodes[node_index].count) / self.nodes[c].count)
            extra = 0
            if len(self.nodes[c].simulation_history) != 0:
                extra = const_2 * math.sqrt(np.var(self.nodes[c].simulation_history) + 1 / self.nodes[c].count)

            result = scaled_reward + exploration_term + extra  # Schadd SP-MCTS added term

            # Compare to running maximum
            if result > opt:
                opt = result
                child = c
        
        chosen_mod = self.modifications[self.nodes[child].modification]
        print(colored("Chosen child's modification: {}".format(chosen_mod), "red"))

        return child


    def default_policy(self, node_index):
        start = node_index
        simulate_env = copy.deepcopy(self.nodes[start].env)
        num_modifications_applied = len(self.env.walls) - len(simulate_env.walls) + len(simulate_env.special) - len(self.env.special)
        mods_left = self.max_layer - num_modifications_applied
        
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

        if reward > self.threshold + 0.5:
            print(colored(a, "red"))
            print(colored(reward, "red"))

            for element in a:
                start = self.add_node(element, start).index

            if reward > self.max_reward:
                self.max_reward = reward
                self.opt_env = copy.deepcopy(simulate_env)
            
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
            print("MCTS insufficient to get {} modifications!".format(self.max_layer))
            return (walk, None)

        else:
            modified = make_env(self.env, walk)
            agent = QAgent(modified)
            agent.qlearn(600, render=False)
            rews = utility(agent)
            return (walk, rews)

    
    def best_observed_choice(self):
        vector = []
        for wall in self.env.walls:
            if wall not in self.opt_env.walls:
                tup = (0, wall[0], wall[1])
                vector.append(tup)

        for cell in self.opt_env.special:
            if cell not in self.env.special:
                tup = (1, cell[0], cell[1])
                vector.append(tup)

        # Training to prevent errors arising from connected training
        agent = QAgent(self.opt_env)
        agent.qlearn(600)
        rews = utility(agent)

        x = max(rews, self.max_reward)

        return (vector, x)


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


if __name__ == "__main__":
    map_to_numpy = np.asarray(map, dtype='c')
    env = TaxiEnv(map_to_numpy)
    tree = Tree(env, max_layer)
    tree.initialize()
    tree.ucb_search(iterations=1000)

    # Store data
    r_dir = os.path.abspath(os.pardir)
    data_dir = os.path.join(r_dir, "data")
    csv_dir = os.path.join(data_dir, "tree_{}.csv".format(tree.max_layer))
    txt_dir = os.path.join(data_dir, "mcts_result_{}.txt".format(tree.max_layer))

    with open(csv_dir, "w", newline='') as file:
        fieldnames = list(tree.info(0).keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for node in tree.nodes:
            writer.writerow(tree.info(node.index))

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