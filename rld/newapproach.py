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
import multiprocessing as mp
from multiprocessing import Manager
import ast
import csv

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

map_to_numpy = np.asarray(map, dtype="c")
env = TaxiEnv(map_to_numpy)

with open("cell_ranking.txt", "r") as f:
    cell_ranking = ast.literal_eval(f.read())

with open("wall_ranking.txt", "r") as f:
    wall_ranking = ast.literal_eval(f.read())

modifications = []
for wall in env.walls:
    modifications.append((0, wall))
for row in range(env.width):
    for col in range(env.length):
        modifications.append((1, (row, col)))

#print(modifications)

def simulate_env(env, num_changes):
    features = []
    mods = random.sample(modifications, k=num_changes)
    result = copy.deepcopy(env)

    for change in mods:
        if change[0] == 0:  # walls
            result = result.transition([change[1]])
        elif change[0] == 1:  # cells
            result.special.append(change[1])
        features.append((change[0], change[1][0], change[1][1]))

    features = sorted(features)
    walls_vector = (len(env.walls) - len(result.walls), 0, 0)
    special_vector = (len(result.special), 0, 0)
    features.append(walls_vector)
    features.append(special_vector)

    info = features[0 : num_changes]
    for element in info:
        category = element[0]
        position = (element[1], element[2])
        if category == 0:  # wall
            for i in range(len(wall_ranking)):
                if wall_ranking[i][0] == position:
                    features.append((category, i, 0))
                    break
        
        elif category == 1:  # cell
            for i in range(len(cell_ranking)):
                if cell_ranking[i][0] == position:
                    features.append((category, i, 0))
                    break

    return [result, features]


def qlearn_as_func(agent, env, number, agents):
    agent.qlearn(600, number=number, render=False)
    agents.append(copy.deepcopy(agent))

data = []

if __name__ == "__main__":
    mp.set_start_method = "spawn"
    num_processes = 10
    processes = []
    manager = Manager()
    agents = manager.list()
    categories = []

    starting_points = []
    for i in range(25):
        env.reset()
        starting_points.append(env.current)


    for iter in range(400):
        print(colored("Data addition round {} begins!".format(iter), "red"))
        for i in range(num_processes):
            results = simulate_env(env, 4)
            modified = results[0]
            categories.append(results[1])
            agent = QAgent(modified)
            p = mp.Process(target=qlearn_as_func, args=(agent, modified, i, agents))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.terminate()

        for i in range(num_processes):
            reward = []
            for j in range(len(starting_points)):
                reward.append(agents[i + iter * num_processes].eval(show=False, fixed=starting_points[j])[1])
            
            average = np.mean(reward)

            data.append((categories[i + iter * num_processes], reward, average))
        

    with open("data_2.csv", "w") as file:
        fieldnames = ["vector", "result", "average"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for element in data:
            simple_dict = {}
            simple_dict["vector"] = element[0]
            simple_dict["result"] = element[1]
            simple_dict["average"] = element[2]
            writer.writerow(simple_dict)

    with open("start_2.txt", "w") as file:
        file.write(str(starting_points))
