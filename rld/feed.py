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

input_data = []
output_data = []
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        vector = ast.literal_eval(row['vector'])
        result = ast.literal_eval(row['result'])
        for index in range(len(vector)):
            #if vector[index] == 1 and index < 6:
            #    vector[index] = 2
            #if vector[index] == 0 and index >= 6:
            #    vector[index] = 1
            if vector[index] == 1 and index >= 6:
                vector[index] = 2

        input_data.append(np.array(vector))
        output_data.append(result)

TEST_SIZE = 0.4
x_train, x_test, y_train, y_test = train_test_split(np.array(input_data), np.array(output_data), test_size=TEST_SIZE)

def get_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_dim, )))
    #model.add(Embedding(15, 25, input_length=input_dim))
    #model.add(Reshape((25 * input_dim, )))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim, activation="linear"))

    model.compile(optimizer = Adam(learning_rate=0.01), loss = "mse")
    return model


input_dim = 15
output_dim = 25
model = get_model(input_dim, output_dim)
n = len(input_data)
train_size = int(n * (1 - TEST_SIZE))
test_size = int(n * TEST_SIZE)

x_train = np.reshape(x_train, (train_size, input_dim))
y_train = np.reshape(y_train, (train_size, output_dim))
model.fit(x_train, y_train, epochs=1000)

y_test = np.reshape(y_test, (test_size, output_dim))
x_test = np.reshape(x_test, (test_size, input_dim))

for i in range(len(x_test)):
    a = np.reshape(x_test[i], (1, input_dim))
    b = np.reshape(y_test[i], (1, output_dim))
    print(a)
    print(model.predict(a), b)

model.evaluate(x_test, y_test, verbose=2)