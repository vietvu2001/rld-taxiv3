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


from taxienv import TaxiEnv

map = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class QAgent():
    def __init__(self, env):
        self.env = env
        self.alpha = 0.15
        self.epsilon = 1
        self.rate = 1
        self.q = {}


    def actions(self, state):
        s = self.env.decode(state)
        break_down = (s[0], s[1])
        if break_down in self.env.special:
            ls = [i for i in range(10)]
            return ls
        else:
            ls = [i for i in range(6)]
            return ls


    def get_q_value(self, state, action):
        if action not in self.actions(state):
            raise ValueError

        if (state, action) not in self.q:
            return 0
        else:
            return self.q[(state, action)]


    def best_reward_state(self, state):
        opt = float("-inf")
        ls = self.actions(state)

        for a in ls:
            if (state, a) in self.q:
                if self.q[(state, a)] > opt:
                    opt = self.q[(state, a)]
            else:
                if opt < 0:  # if (state, a) not in self.q, we default a q-value of 0
                    opt = 0

        return opt


    def choose_action(self, state, prob = True):

        values = [0, 1]
        probs = [self.epsilon, 1 - self.epsilon]
        rand = random.choices(values, probs)
        ls = self.actions(state)

        if not prob or rand == 1:
            opt = float("-inf")
            action = None
            for a in ls:
                value = self.get_q_value(state, a)
                if value > opt:
                    opt = value
                    action = a
            return action

        else:
            action = random.choice(ls)
            return action


    def qlearn(self, num_episodes, show=True, number=None, render=True):

        for i in range(num_episodes):
            if show and i % 10 == 0 or i == num_episodes - 1:
                if number is None:
                    print("Episode {} begins!".format(i))
                else:
                    print("Episode {} begins! ({})".format(i, number))

            s = self.env.reset()
            t = 0

            while True and t < 3000:
                action = self.choose_action(s)
                s_next, reward, done = self.env.step(action)

                future_rewards_estimated = self.best_reward_state(s_next)
                old_q = self.get_q_value(s, action)

                # Update q-value
                self.q[(s, action)] = old_q + self.alpha * (reward + self.rate * future_rewards_estimated - old_q)

                s = s_next
                if done:
                    break

                t += 1

            if show and i % 10 == 0 or i == num_episodes - 1:
                if number is None:
                    print("Episode {} done ({})!".format(i, t))
                else:
                    print("Episode {} done ({})({})!".format(i, t, number))
                
                if render:
                    self.env.render()

            self.epsilon *= 0.995


    def eval(self, show=True, fixed=None):
        s = self.env.reset()
        if fixed is not None:
            self.env.current = fixed
            s = self.env.current
        if show:
            self.env.render()
        steps = []
        bd = self.env.decode(s)
        states = [(bd[0], bd[1])]
        t = 0
        total = 0

        while t < 1000:
            action = self.choose_action(s, prob = False)
            s_next, reward, done = self.env.step(action)
            steps.append(action)
            break_down = self.env.decode(s_next)
            states.append((break_down[0], break_down[1]))
            total += reward

            s = s_next
            t += 1

            if done:
                break

        if show:
            self.env.render()
        return [steps, total, states]


'''map_to_numpy = np.asarray(map, dtype='c')
env = TaxiEnv(map_to_numpy)
agent = QAgent(env)
start = time.time()
agent.qlearn(600)
end = time.time()
print(end - start)
starting_points = []
for i in range(25):
    env.reset()
    starting_points.append(env.current)

print(starting_points)
for i in range(2):
    print(agent.eval(fixed=223))'''
