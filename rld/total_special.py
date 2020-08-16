import numpy as np  # pylint: disable=import-error
import random
from gym import utils  # pylint: disable=import-error
from io import StringIO
import sys
from contextlib import closing
import copy

import numpy as np  # pylint: disable=import-error
import gym  # pylint: disable=import-error
import time
import random
from collections import deque

import tensorflow as tf  # pylint: disable=import-error
from tensorflow import keras  # pylint: disable=import-error
from keras.layers import Reshape, BatchNormalization  # pylint: disable=import-error
from keras.layers.embeddings import Embedding  # pylint: disable=import-error

num_inputs = 1
num_actions = 6
num_hidden = 64


map = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

gamma = 1  # Discount factor for past rewards
max_steps_per_episode = 3000
eps = 1e-6

class TaxiEnv():
    def __init__(self, map_encode_by_numpy):
        self.length = 5
        self.width = 5
        self.max_row = self.width - 1
        self.max_col = self.length - 1

        # Destinations
        self.dest = [(0, 0), (0, 4), (4, 0), (4, 3)]

        # Number of states
        self.num_states = 500

        # numpy encoding of map
        self.desc = map_encode_by_numpy

        dims = self.desc.shape
        self.walls = []
        self.non_walls = []
        
        for i in range(1, dims[0] - 1):
            for j in range(1, dims[1] - 1):
                if self.desc[i][j] == b"|":
                    self.walls.append((i, j))
                elif self.desc[i][j] == b":":
                    self.non_walls.append((i, j))

        self.reward_cells = []
        self.current = None
        self.special = []
        

    def terminal(self, state):
        if state in self.dest:
            return True
        return False
    
    # Represent state: (row, col, pass_id, dest_id)
    # id: 0 - (0, 0), 1 - (0, 4), 2 - (4, 0), 3 - (4, 3)

    def step(self, action):

        assert self.current != None

        state = self.decode(self.current)

        taxi_loc = (state[0], state[1])
        new_row, new_col, new_pass_idx, dest_id = state
        done = False
        reward = -1

        if action == 0:
            new_row = min(state[0] + 1, self.max_row)
        elif action == 1:
            new_row = max(state[0] - 1, 0)
        elif action == 2 and self.desc[1 + state[0], 2 * state[1] + 2] == b":":
            new_col = min(state[1] + 1, self.max_col)
        elif action == 3 and self.desc[1 + state[0], 2 * state[1]] == b":":
            new_col = max(state[1] - 1, 0)
        elif action == 4:  # pickup
            if (state[2] < 4 and taxi_loc == self.dest[state[2]]):
                new_pass_idx = 4
            else: # passenger not at location
                reward = -10
        elif action == 5:  # dropoff
            if (taxi_loc == self.dest[state[3]]) and state[2] == 4:
                new_pass_idx = state[3]
                done = True
                reward = 20
            elif (taxi_loc in self.dest) and state[2] == 4:
                new_pass_idx = self.dest.index(taxi_loc)
            else: # dropoff at wrong location
                reward = -10
        elif len(self.special) != 0 and taxi_loc in self.special:
            if action == 6:
                propo = self.desc[state[0], 2 * state[1] + 2] == b":" or self.desc[state[0] + 1, 2 * state[1] + 2] == b":"
                if propo and state[0] >= 1 and state[1] + 1 <= self.max_col:
                    new_row = state[0] - 1
                    new_col = state[1] + 1
                
            elif action == 7:
                propo = self.desc[state[0] + 1, 2 * state[1] + 2] == b":" or self.desc[state[0] + 2, 2 * state[1] + 2] == b":"
                if propo and state[0] + 1 <= self.max_row and state[1] + 1 <= self.max_col:
                    new_row = state[0] + 1
                    new_col = state[1] + 1
            
            elif action == 8:
                propo = self.desc[state[0] + 1, 2 * state[1]] == b":" or self.desc[state[0] + 2, 2 * state[1]] == b":"
                if propo and state[0] + 1 <= self.max_row and state[1] >= 1:
                    new_row = state[0] + 1
                    new_col = state[1] - 1
            
            elif action == 9:
                propo = self.desc[state[0], 2 * state[1]] == b":" or self.desc[state[0] + 1, 2 * state[1]] == b":"
                if propo and state[0] >= 1 and state[1] >= 1:
                    new_row = state[0] - 1
                    new_col = state[1] - 1
            
        
        next_state = (new_row, new_col, new_pass_idx, dest_id)
        next_state = self.encode(next_state)
        self.current = next_state
        return (next_state, reward, done)
    
    def encode(self, state):
    # (5) 5, 5, 4
        i = state[0]
        i *= 5
        i += state[1]
        i *= 5
        i += state[2]
        i *= 4
        i += state[3]
        return i
    
    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return tuple(reversed(out))
    
    def reset(self):
        row = random.randint(0, 4)
        col = random.randint(0, 4)
        pass_id = random.randint(0, 3)
        dest_id = random.randint(0, 3)
        while dest_id == pass_id:
            dest_id = random.randint(0, 3)
        state = self.encode((row, col, pass_id, dest_id))
        self.current = state
        return state
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.current)

        def ul(x): return "_" if x == " " else x
        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.dest[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        di, dj = self.dest[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
    
    # Simple transition function: remove 2 random walls
    def transition(self, walls_to_cut=None):
        if walls_to_cut is None:
            alternate = copy.deepcopy(self)
            walls = random.sample(self.walls, 1)
            for wall in walls:
                wall = tuple(wall)
                alternate.walls.remove(wall)
                alternate.desc[wall[0]][wall[1]] = b":"
            return alternate
        else:
            alternate = copy.deepcopy(self)
            for wall in walls_to_cut:
                wall = tuple(wall)
                alternate.walls.remove(wall)
                alternate.desc[wall[0]][wall[1]] = b":"
            return alternate


def worker(num, actor, critic, env):
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    under_20 = 0

    books = deque(maxlen=300)

    while True:  # Run until solved
        state = env.reset()
        episode_reward = 0
        penalties = 0
        drop = 0
        print("Episode {} begins".format(episode_count))
        env.render()
        start = time.time()

        time_solve = 0

        with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
        #with tf.GradientTape() as tape:
        #while True:
            for _ in range(1, max_steps_per_episode + 1):
            #env.render()  # Adding this line would show the attempts
            # of the agent in a pop up window.

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs = actor(state)
                critic_value = critic(state)
                critic_value_history.append((state, critic_value[0, 0]))

                # Choose action
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action])) # action_probs stores log of probs of action

                #if timestep == 1:
                #  print("{}: {}".format(state, action_probs))
                #  print("{}: {}".format(state, action))

                # Apply the sampled action in our environment
                state, reward, done = env.step(action)
                rewards_history.append(reward)
                episode_reward += reward
                time_solve += 1

                if reward == -10:
                    penalties += 1
            
                elif reward == 20:
                    drop += 1

                if done:
                    break
        
            # Update running reward to check condition for solving
            books.appendleft(episode_reward)
            running_reward = sum(books) / books.maxlen

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = deque(maxlen=3500)
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.appendleft(discounted_sum)

            # Normalize
            #returns = np.array(returns)
            #returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            #returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            loss_value_actor = 0
            loss_value_critic = 0
            for log_prob, value, ret in history:
                diff = ret - value[1]
                loss_value_actor += -log_prob * diff
                loss_value_critic += huber_loss(tf.expand_dims(value[1], 0), tf.expand_dims(ret, 0))

            # Backpropagation
            loss_value_actor /= time_solve
            loss_value_critic /= time_solve
        
            if episode_count % 2 == 1:
                grads_1 = tape_1.gradient(loss_value_actor, actor.trainable_variables)
                optimizer.apply_gradients(zip(grads_1, actor.trainable_variables))
        
            grads_2 = tape_2.gradient(loss_value_critic, critic.trainable_variables)
            optimizer.apply_gradients(zip(grads_2, critic.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # Log details
        end = time.time()
        episode_count += 1
        if episode_count % 1 == 0:
            env.render()
            template = "average reward: {:.2f}"
            print(template.format(running_reward, episode_count))
            print("episode reward: {}".format(episode_reward))
            print("Steps taken: {}".format(time_solve))
            print("Penalties incurred: {}".format(penalties))
            print("Passengers dropped off: {}".format(drop))
            print("Time taken: {}".format(end - start))
            print()

        if running_reward > 2:  # Condition to consider the task solved
            under_20 += 1
    
        if under_20 > 5:
            print("Solved at episode {} !".format(episode_count))
            break

def evaluate(actor, critic, env, inp=None, show=True):  # greedy
    if inp is not None:
        env.reset()
        env.current = inp
        state = env.current
    else:
        state = env.reset()
    actions = []
    states = []
    states_full = []
    break_down = env.decode(state)
    states.append((break_down[0], break_down[1]))
    states_full.append(break_down)
    if show:
        env.render()
    for _ in range(1, 2000):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probs = actor(state)
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))

        state, _, done = env.step(action)
        actions.append(action)
        break_down = env.decode(state)
        states.append((break_down[0], break_down[1]))
        states_full.append(break_down)

        if done:
            break
    return (actions, states, states_full)


def cell_frequency(actor, critic, env, iter=10):
    dict_return = {}
    for row in range(env.width):
        for col in range(env.length):
            dict_return[(row, col)] = 0

    for _ in range(iter):
        states = evaluate(actor, critic, env, show=False)[1]
        for state in states:
            dict_return[state] += 1

    dict_return = sorted(dict_return.items(), key=lambda x: -x[1])
    return dict_return


def wall_interference(actor, critic, env, iter=10):
    dict_return = {}
    for _ in range(iter):
        a = evaluate(actor, critic, env, show=False)
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

#map_to_numpy = np.asarray(map, dtype='c')
#env = TaxiEnv(map_to_numpy)
#actor = keras.models.load_model("my_actor")
#critic = keras.models.load_model("my_critic")
#print(cell_frequency(actor, critic, env, 1000))
#print(wall_interference(actor, critic, env, 1000))
#for _ in range(400):
#    print(evaluate(actor, critic, env, 204))
        