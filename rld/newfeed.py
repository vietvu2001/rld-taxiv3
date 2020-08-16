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
import csv
import ast

import tensorflow as tf  # pylint: disable=import-error
from tensorflow import keras  # pylint: disable=import-error
from tensorflow.keras import layers  # pylint: disable=import-error
from keras.layers import Reshape, BatchNormalization, Dense, Dropout, Embedding  # pylint: disable=import-error
from keras.layers.embeddings import Embedding  # pylint: disable=import-error
from keras.models import Sequential  # pylint: disable=import-error
from keras.optimizers import Adam  # pylint: disable=import-error
from keras.losses import MeanAbsoluteError  # pylint: disable=import-error

from sklearn.model_selection import train_test_split  # pylint: disable=import-error
from taxienv import TaxiEnv
from image import map_to_colors
import matplotlib.pyplot as plt  # pylint: disable=import-error

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
input_data = []
output_data = []

with open("data_2.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        vector = ast.literal_eval(row['vector'])
        result = ast.literal_eval(row['average'])

        input_data.append(vector)
        #print(vector, transform_vector)
        output_data.append(result)


def convert_to_image(data, env_ref):
    orig = copy.deepcopy(env_ref)
    assert len(orig.walls) == 6
    assert len(orig.special) == 0

    positions = data[0 : 4]
    for element in positions:
        if element[0] == 0:
            orig = orig.transition([(element[1], element[2])])
        elif element[0] == 1:
            orig.special.append((element[1], element[2]))
    
    img = map_to_colors(orig)
    return img


def process_input(input, env_ref):
    images = []
    n = len(input)
    for i in range(n):
        img = convert_to_image(input[i], env_ref)
        images.append(img)

    return images

def process_output(output):
    labels = []
    space_ref = [9 + 0.1 * i for i in range(11)]
    n = len(output)
    for i in range(n):
        a = output[i]
        if space_ref[0] > a:
            labels.append(0)
        elif space_ref[len(space_ref) - 1] <= a:
            labels.append(len(space_ref))
        else:
            for j in range(len(space_ref) - 1):
                if space_ref[j] <= a < space_ref[j + 1]:
                    labels.append(j + 1)
                    break

    return labels


def get_model(IMG_WIDTH, IMG_LENGTH):
    model = tf.keras.models.Sequential([
        # Convolution layer 1: Input
        tf.keras.layers.Conv2D(
            10, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_LENGTH, 3)
        ),

        # Batch Normalization
        tf.keras.layers.BatchNormalization(axis=1),

        # Pooling layer 1
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add hidden layers
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dense(512, activation = "relu"),

        # Last batch normalization
        tf.keras.layers.BatchNormalization(),

        # Add output layer
        tf.keras.layers.Dense(12, activation = "softmax")
    ])

    model.compile(optimizer = "adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model


collection = process_input(input_data, env)
labels = process_output(output_data)
model = get_model(5, 9)

TEST_SIZE = 0.2
labels = tf.keras.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(
        np.array(collection), np.array(labels), test_size=TEST_SIZE
    )

model.fit(x_train, y_train, epochs=100)