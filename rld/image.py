import cv2  # pylint: disable=import-error
import numpy as np  #pylint: disable=import-error
from PIL import Image  #pylint: disable=import-error
import matplotlib.pyplot as plt  # pylint: disable=import-error

from taxienv import TaxiEnv
import copy

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

#print(env.desc.shape)


def map_to_colors(env):
    COLORS = {b' ': [255, 255, 255], b':': [0, 255, 0], b'|': [0, 0, 0], b'R': [216, 30, 54], b'+': [0, 0, 0], b'-': [0, 0, 0], b'G': [204, 0, 204],
            b'B': [2, 81, 154], b'Y': [238, 223, 16]}

    color_map = np.zeros([env.desc.shape[0] - 2, env.desc.shape[1] - 2, 3], dtype=int)
    for i in range(env.desc.shape[0] - 2):
        for j in range(env.desc.shape[1] - 2):
            color_map[i][j] = np.array(copy.deepcopy(COLORS[env.desc[i + 1][j + 1]]))
    
    for i in range(env.width):
        for j in range(env.length):
            if (i, j) in env.special:
                color_map[i][2 * j] = np.array([254, 151, 0])

    return color_map

#modified = copy.deepcopy(env)
#modified = modified.transition([(2, 4)])
#modified.special.append((2, 1))
#a = map_to_colors(modified)
#print(a)
#plt.imshow(a, interpolation="nearest")
#plt.show()