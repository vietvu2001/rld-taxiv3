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

modifications = []
for wall in env.walls:
    modifications.append(('w', wall))
for row in range(env.width):
    for col in range(env.length):
        modifications.append(('c', (row, col)))

#print(modifications)

def simulate_env(env, num_changes):
    one_hot = [0] * len(modifications)
    m = []
    mods = random.sample(modifications, k=num_changes)
    result = copy.deepcopy(env)
    for change in mods:
        if change[0] == "w":
            result = result.transition([change[1]])
        elif change[0] == "c":
            result.special.append(change[1])

    for element in mods:
        index = modifications.index(element)
        one_hot[index] = 1
        m.append(index)

    return [result, m]


def qlearn_as_func(agent, env, number, agents):
    agent.qlearn(600, number=number)
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


    for iter in range(40):
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
            print(starting_points)
            reward = []
            for j in range(len(starting_points)):
                reward.append(agents[i + iter * num_processes].eval(show=False, fixed=starting_points[j])[1])
            
            average = np.mean(reward)

            data.append((categories[i + iter * num_processes], reward, average))
        
        time.sleep(1)

    with open("data_1.csv", "w") as file:
        fieldnames = ["vector", "result", "average"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for element in data:
            simple_dict = {}
            simple_dict["vector"] = element[0]
            simple_dict["result"] = element[1]
            simple_dict["average"] = element[2]
            writer.writerow(simple_dict)

    with open("start_1.txt", "w") as file:
        file.write(str(starting_points))


    
