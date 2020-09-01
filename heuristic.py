import csv
import ast
import numpy as np  # pylint: disable=import-error
import copy
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

map_to_numpy = np.asarray(map, dtype="c")
env = TaxiEnv(map_to_numpy)  # reference environment

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
            ref_env = ref_env.transition([(element[1], element[2])])
        else:
            ref_env.special.append((element[1], element[2]))
    
    return ref_env


def cell_frequency(agent):
    dict_return = {}
    for row in range(env.width):
        for col in range(env.length):
            if (row, col) not in agent.env.dest:
                dict_return[(row, col)] = 0

    ls = agent.env.resettable_states()
    for i in range(len(ls)):
        states = agent.eval(show=False, fixed=ls[i])[2]
        for state in states:
            if state not in agent.env.dest:
                dict_return[state] += 1

    dict_return = sorted(dict_return.items(), key=lambda x: -x[1])
    return dict_return


def evaluate(agent, show=True, fixed=None):
    s = agent.env.reset()
    if fixed is not None:
        agent.env.current = fixed
        s = agent.env.current
    if show:
        agent.env.render()
    steps = []
    bd = agent.env.decode(s)
    states = [(bd[0], bd[1])]
    states_full = [bd]
    t = 0
    total = 0

    while t < 1000:
        action = agent.choose_action(s, prob = False)
        s_next, reward, done = agent.env.step(action)
        steps.append(action)
        break_down = agent.env.decode(s_next)
        states.append((break_down[0], break_down[1]))
        states_full.append(break_down)
        total += reward

        s = s_next
        t += 1

        if done:
            break

        if show:
            agent.env.render()

    return [steps, states, states_full]


def wall_interference(agent):
    dict_return = {}
    ls = agent.env.resettable_states()

    for i in range(len(ls)):
        a = evaluate(agent, show=False, fixed=ls[i])
        t = 0
        while (a[2][t][2] == a[2][t + 1][2]):
            t += 1

        b = a[1][0 : (t + 1)]
        for pos in range(len(b) - 1):
            row = b[pos][0]
            r = pos + 1
            while r < len(b):
                if b[pos][0] != b[r][0]:
                    r += 1
                else:
                    break

            if r != len(b):
                steps = r - pos
                if steps > abs(b[r][1] - b[pos][1]) and b[r] != b[pos]:  # there is a wall
                    p_1 = 2 * b[pos][1] + 1
                    p_2 = 2 * b[r][1] + 1
                    for i in range(min(p_1, p_2), max(p_1, p_2) + 1):
                        if env.desc[row + 1, i] == b"|":
                            if (row + 1, i) not in dict_return:
                                dict_return[(row + 1, i)] = 1
                            else:
                                dict_return[(row + 1, i)] += 1

        c = a[1][(t + 1) : len(a[1])]
        for pos in range(len(c) - 1):
            row = c[pos][0]
            r = pos + 1
            while r < len(c):
                if c[pos][0] != c[r][0]:
                    r += 1
                else:
                    break

            if r != len(c):
                steps = r - pos
                if steps > abs(c[r][1] - c[pos][1]) and c[r] != c[pos]:  # there is a wall
                    p_1 = 2 * c[pos][1] + 1
                    p_2 = 2 * c[r][1] + 1
                    for i in range(min(p_1, p_2), max(p_1, p_2) + 1):
                        if env.desc[row + 1, i] == b"|":
                            if (row + 1, i) not in dict_return:
                                dict_return[(row + 1, i)] = 1
                            else:
                                dict_return[(row + 1, i)] += 1
    
    dict_return = sorted(dict_return.items(), key=lambda x: -x[1])
    return dict_return


def path_based(env, num_mods):
    # Train agent in original environment
    agent = QAgent(env)
    agent.qlearn(600, render=False)
    cell_dict = cell_frequency(agent)
    wall_dict = wall_interference(agent)

    opt_seq = None
    opt_val = float("-inf")
    
    for k in range(num_mods):
        seq = []
        ref = copy.deepcopy(env)
        # Pick k modifications on walls
        walls_to_remove = [wall_dict[i][0] for i in range(k)]
        for wall in walls_to_remove:
            ref = ref.transition([(wall)])
            seq.append((0, wall[0], wall[1]))

        num_special_cells = num_mods - k
        cells_to_assign = [cell_dict[i][0] for i in range(num_special_cells)]

        for cell in cells_to_assign:
            ref.special.append(cell)
            seq.append((1, cell[0], cell[1]))
        
        agent_k = QAgent(ref)
        print(colored("Iteration {} begins!".format(k), "red"))
        print(ref.walls, ref.special)
        agent_k.qlearn(600, render=False)

        rews = utility(agent_k)
        if rews > opt_val:
            opt_val = rews
            opt_seq = seq

    return (opt_seq, opt_val)