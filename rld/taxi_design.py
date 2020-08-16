import numpy as np  # pylint: disable=import-error
import random
from gym import utils  # pylint: disable=import-error
from io import StringIO
import sys
from contextlib import closing
import copy

import gym  # pylint: disable=import-error
import time
from collections import deque

import tensorflow as tf  # pylint: disable=import-error
from tensorflow import keras  # pylint: disable=import-error
from tensorflow.keras import layers  # pylint: disable=import-error
from keras.layers import Reshape, BatchNormalization  # pylint: disable=import-error
from keras.layers.embeddings import Embedding  # pylint: disable=import-error

num_inputs = 1
num_actions = 6
num_hidden = 75

def build_actor_network_branch(num_inputs, num_hidden, num_actions):
    import tensorflow as tf  # pylint: disable=import-error
    from tensorflow import keras  # pylint: disable=import-error
    from tensorflow.keras import layers  # pylint: disable=import-error
    from keras.layers import Reshape, BatchNormalization  # pylint: disable=import-error
    from keras.layers.embeddings import Embedding  # pylint: disable=import-error
    
    inputs_1 = layers.Input(shape=(num_inputs,))
    embed = layers.Embedding(500, 10, input_length=num_inputs)(inputs_1)
    reshape = layers.Reshape((10 * num_inputs, ))(embed)
    common = layers.Dense(num_hidden * 2, activation="relu")(reshape)
    common = layers.Dense(num_hidden, activation="relu")(common)
    action = layers.Dense(num_actions, activation="softmax")(common)
    action_special = layers.Dense(10, activation="softmax")(common)

    model_1 = keras.Model(inputs=inputs_1, outputs=[action, action_special])
    return model_1


# Critic Reward Network
def build_critic_network_branch(num_inputs, num_hidden):
    import tensorflow as tf  # pylint: disable=import-error
    from tensorflow import keras  # pylint: disable=import-error
    from tensorflow.keras import layers  # pylint: disable=import-error
    from keras.layers import Reshape, BatchNormalization  # pylint: disable=import-error
    from keras.layers.embeddings import Embedding  # pylint: disable=import-error

    huber_loss = keras.losses.Huber()

    inputs_2 = layers.Input(shape=(num_inputs,))
    embed_2 = layers.Embedding(500, 10, input_length=num_inputs)(inputs_2)
    reshape_2 = layers.Reshape((10, ))(embed_2)
    common_2 = layers.Dense(num_hidden * 2, activation="relu")(reshape_2)
    common_2 = layers.Dense(num_hidden, activation="relu")(common_2)
    critic = layers.Dense(1)(common_2)

    model_2 = keras.Model(inputs=inputs_2, outputs=critic)
    model_2.compile(optimizer = keras.optimizers.Adam(learning_rate=3e-4), loss=huber_loss)
    return model_2


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

from taxienv import TaxiEnv

def worker(num, actor, critic, env, artf=False):
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    great = 0

    books = deque(maxlen=100)

    while True:  # Run until solved
        if artf:
            state = env.reset()
            env.current = env.encode((0, 0, 1, 3))
            state = env.current
        else:
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

                s = env.decode(state)
                s_coord = (s[0], s[1])

                check = s_coord in env.special

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                if check:
                    action_probs = actor(state)[1]
                else:
                    action_probs = actor(state)[0]

                critic_value = critic(state)
                critic_value_history.append((state, critic_value[0, 0]))

                # Choose action
                if check:
                    action = np.random.choice(10, p=np.squeeze(action_probs))
                    action_probs_history.append(tf.math.log(action_probs[0, action])) # action_probs stores log of probs of action
                
                else:
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

            print(loss_value_actor)
        
            if episode_count % 2 == 1:
                grads_1 = tape_1.gradient(loss_value_actor, actor.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads_1, actor.trainable_variables) if grad is not None)
        
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

        if running_reward > 8.7:  # Condition to consider the task solved
            great += 1
    
        if great > 20:
            print("Solved at episode {} !".format(episode_count))
            break

def evaluate_branch(actor, critic, env, inp=None, show=True):  # greedy
    if inp is not None:
        env.reset()
        env.current = inp
        state = env.current
    else:
        state = env.reset()
    actions = []
    states = []
    break_down = env.decode(state)
    states.append((break_down[0], break_down[1]))
    if show:
        env.render()
    for _ in range(1, 2000):
        s = env.decode(state)
        s_coord = (s[0], s[1])
        check = s_coord in env.special
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action = -1

        if check:
            action_probs = actor(state)[1]
            action = np.random.choice(10, p=np.squeeze(action_probs))
        
        else:
            action_probs = actor(state)[0]
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

        state, _, done = env.step(action)
        actions.append(action)
        break_down = env.decode(state)
        states.append((break_down[0], break_down[1]))

        if done:
            break
    
    if 6 in actions or 7 in actions or 8 in actions or 9 in actions:
        if show:
            print("Special move made!")
        
    return (actions, states)

def cell_frequency(actor, critic, env, iter=10):
    dict_return = {}
    for _ in range(iter):
        states = evaluate_branch(actor, critic, env, show=False)[1]
        for state in states:
            if state not in env.dest:
                if state not in dict_return:
                    dict_return[state] = 1
                else:
                    dict_return[state] += 1

    dict_return = sorted(dict_return.items(), key=lambda x: -x[1])
    return dict_return


#map_to_numpy = np.asarray(map, dtype='c')
#env = TaxiEnv(map_to_numpy)
#env = env.transition([(4, 2), (2, 4)])
#env.special.append((2, 1))
#env.special.append((2, 2))

#actor = build_actor_network_branch(num_inputs, num_hidden, num_actions)
#critic = build_critic_network_branch(num_inputs, num_hidden)
#actor = keras.models.load_model("my_actor_freq_2")
#critic = keras.models.load_model("my_critic_freq_2")
#worker(0, actor, critic, env)

#actor.save("my_actor_freq_2")
#critic.save("my_critic_freq_2")
#for _ in range(400):
#    print(evaluate_branch(actor, critic, env))
#print(cell_frequency(actor, critic, env, 500))