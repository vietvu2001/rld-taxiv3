/*
import numpy as np  # pylint: disable=import-error
import random
from gym import utils  # pylint: disable=import-error
from io import StringIO
import sys
from contextlib import closing
import copy

import gym  # pylint: disable=import-error
from termcolor import colored  # pylint: disable=import-error
import time
import random
from collections import deque
*/


//from wgrenv import WindyGridworld

#include "w_qlearn.h"

w_QAgent::w_QAgent(self, env) {
    this->env = env;
    this->alpha = 0.5;
    this->epsilon = 1;
    this->rate = 1;
    this->q = {};
}


std::vector<int> w_QAgent::actions(std::vector<int> state) {
    bool special_cell = false;
    bool jump_cell = false;

    for (auto & it : this->env.special) {
        std::pair<int, int> tuple;
        tuple.first = state[0];
        tuple.second = state[1];

        if (tuple == *it) {
            special_cell = true;
        }
    }
    
    for (auto & it : this->env.jump_cells) {
        std::pair<int, int> tuple;
        tuple.first = state[0];
        tuple.second = state[1];

        if (tuple == *it) {
            jump_cell = true;
        }
    }

    if (special_cell && jump_cell) {

        std::vector<int> ls;
        for (int i = 0; i < 12; ++i) {
            ls.push_back(i);
        }
        return ls;

    } else if (special_cell) {
        std::vector<int> ls;
        for (int i = 0; i < 8; ++i) {
            ls.push_back(i);
        }
        return ls;
    } else if (jump_cell) {
        std::vector<int> ls;
        for (int i = 0; i < 4; ++i) {
            ls.push_back(i);
        }

        for (int i = 8; i < 12; ++i) {
            ls.push_back(i);
        }

        return ls;
    } else {
        std::vector<int> ls;
        for (int i = 0; i < 4; ++i) {
            ls.push_back(i);
        }

    }
}


int w_QAgent::get_q_value(int state, float action) {
        if action not in self.actions(state):
            raise ValueError

        if state not in self.q:
            return 0

        elif action not in self.q[state]:
            return 0

        else:
            return self.q[state][action]
}


w_QAgent::best_reward_state(self, state):
        opt = float("-inf")
        ls = self.actions(state)

        if state not in self.q:
            return 0

        for a in ls:
            if a in self.q[state]:
                if self.q[state][a] > opt:
                    opt = self.q[state][a]
            else:
                if opt < 0:  # if (state, a) not in self.q, we default a q-value of 0
                    opt = 0

        return opt


w_QAgent::choose_action(self, state, prob = True):
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


w_QAgent::qlearn(self, num_episodes, show=True, number=None, render=True):

        for i in range(num_episodes):
            if show and (i % 100 == 0 or i == num_episodes - 1):
                if number is None:
                    print("Episode {} begins!".format(i))
                else:
                    print("Episode {} begins! ({})".format(i, number))

            s = self.env.reset()
            t = 0

            while True and t < 2000:
                action = self.choose_action(s)
                s_next, reward, done = self.env.step(action)

                future_rewards_estimated = self.best_reward_state(s_next)
                old_q = self.get_q_value(s, action)

                # Update q-value
                if s not in self.q:
                    self.q[s] = {}

                self.q[s][action] = old_q + self.alpha * (reward + self.rate * future_rewards_estimated - old_q)

                s = s_next
                if done:
                    break

                t += 1

            if show and (i % 100 == 0 or i == num_episodes - 1):
                if number is None:
                    print("Episode {} done ({})!".format(i, t))
                else:
                    print("Episode {} done ({})({})!".format(i, t, number))
                
                if render:
                    self.env.render()

            if self.epsilon >= 0.1:
                self.epsilon *= 0.995


w_QAgent::eval(self, show=True, fixed=None):
        s = self.env.reset()
        if fixed is not None:
            self.env.current = fixed
            s = self.env.current
        if show:
            self.env.render()
        steps = []

        states = [s]
        t = 0
        total = 0

        while t < 1000:
            action = self.choose_action(s, prob = False)
            s_next, reward, done = self.env.step(action)
            steps.append(action)

            states.append(s_next)
            total += reward

            s = s_next
            t += 1

            if done:
                break

        if show:
            self.env.render()
        return [steps, total, states]


w_QAgent::print_eval_result(self, output):
        # Print results from evaluation for visualization
        print("Steps taken: {}".format(output[0]))
        print("Total reward: {}".format(output[1]))
        print("States traversed: {}".format(output[2]))

        return

int main() {
    env = WindyGridworld()

    modified = copy.deepcopy(env)
    modified.jump_cells.append((4, 1))
    modified.jump_cells.append((4, 3))
    modified.special.append((3, 5))

    start = time.time()
    agent = w_QAgent(modified)
    agent.qlearn(3000, render=False)
    end = time.time()

    series = env.resettable_states()
    vals = []

    for state in series:
        res = agent.eval(fixed=state, show=False)
        vals.append(res[1])

    print(colored(np.mean(vals), "red"))
    agent.env.render()

    print(colored(end - start, "red"))

}