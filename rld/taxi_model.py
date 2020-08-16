import numpy as np  # pylint: disable=import-error
import gym  # pylint: disable=import-error
import time
import random
from collections import deque

import tensorflow as tf  # pylint: disable=import-error
from tensorflow import keras  # pylint: disable=import-error
from tensorflow.keras import layers  # pylint: disable=import-error
from keras.layers import Reshape, BatchNormalization  # pylint: disable=import-error
from keras.layers.embeddings import Embedding  # pylint: disable=import-error

import ray  # pylint: disable=import-error
import multiprocessing as mp

# Configuration parameters for the whole setup
gamma = 1  # Discount factor for past rewards
max_steps_per_episode = 2500
env = gym.make("Taxi-v3").env  # Create the environment
eps = 1e-6

num_inputs = 1
num_actions = 6
num_hidden = 64

simulated_epsilon = 0

# Actor Policy Network
def build_actor_network(num_inputs, num_hidden, num_actions):
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

    model_1 = keras.Model(inputs=inputs_1, outputs=action)
    return model_1


# Critic Reward Network
def build_critic_network(num_inputs, num_hidden):
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

model_1 = keras.models.load_model("my_actor")
model_2 = keras.models.load_model("my_critic")

def worker(num, actor, critic):
    import tensorflow as tf  # pylint: disable=import-error
    from tensorflow import keras  # pylint: disable=import-error
    from tensorflow.keras import layers  # pylint: disable=import-error
    from keras.layers import Reshape, BatchNormalization  # pylint: disable=import-error
    from keras.layers.embeddings import Embedding  # pylint: disable=import-error
    
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    under_20 = 0

    books = deque(maxlen=1000)

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
                state, reward, done, _ = env.step(action)
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
            running_reward = sum(books) / 1000

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


def evaluate(actor, critic):  # greedy
    env = gym.make("Taxi-v3").env
    state = env.reset()
    actions = []
    env.render()
    for _ in range(1, 2000):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_probs = actor(state).numpy()
        action = np.argmax(action_probs)
        state, _, done, _ = env.step(action)
        actions.append(action)

        if done:
            break
    return actions


