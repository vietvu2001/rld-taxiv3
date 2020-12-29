This markdown contains the aims and instructions for running each Python file in ```src/```.

1. ```allstates.py```

- This file collects the training data needed for Supervised Learning. We create environments using random vectors of modifications (fixed number of iterations), and evaluate them using Q-Learning agents. To specify how many modifications are allowed, modify line ```95```, along with line ```85``` to figure out how many rounds you want to do (based on the number of environments you want to generate, which is clearly the product of ```rounds``` and ```num_processes``` in the file).

- The data we store consists of the vector of modifications, and the utility of a Q-Learning agent in that environment. Here utility is defined as the average taken over the total rewards from all possible starting positions. This data is stored in ```data/data_X.csv``` (where X is the number of modifications).

- Usage: ```python allstates.py```

2. ```feed.py```

- This file reads the data in ```data/data_X.csv``` and performs training of a model, in which the input is the random vector of modifications and the output is the utility. The random vector of modifications is flattened. 
- We use this model to predict the utility of a Q-Learning agent in a large number of environments. You can tune this number on lines ```273``` and ```276```. We build a minimum heap to store a small number of environments with highest predictions (defined as ```12 * num_mods```), and do Q-Learning evaluations on them to get exact numbers for comparison.
- The output contains the chosen vector of modifications, along with the corresponding utility. It is stored in ```data/sl_result_X.txt``` (where X is the number of modifications).

- Usage: ```python feed.py```

3. ```heuristic.py```

- This file contains the implementations of the ```cell_frequency``` and ```wall_interference``` heuristics. The input is an agent of class ```Agent``` (as stored in ```qlearn.py```), in which its environment is a field. We return the rankings of the cells that the agent crosses the most, the the rankings of the walls for whose elimination will improve the rewards from several starting states (rank by how many starting states can be improved). 

- We also implement the ```utility``` function, which also takes an agent of class ```Agent``` as input, and returns the utility of the agent in its respective environment.

- This file is used as a helper file.

4. ```qlearn.py```

- This file implements the training and evaluation of a Q-Learning agent in its respective environment. The class ```Agent``` is implemented in this file. It is used in the other files as a helper file, but there is no output in this file.

5. ```mcts.py```

- This file implements the Monte Carlo Tree Search algorithm for the problem. The number of modifications corresponds to the maximum layer of the tree, and denoted as ```max_layer``` in the file. Thus, to modify the number of modifications, modify the value of ```max_layer``` on line ```32```. The number of iterations on which the tree is explored can be modified on line ```415```.

- A greedy search through the tree is conducted. Starting from the root, we choose the best **explored** child based on the UCB heuristic. This is done until we reach the end of the tree (as in reaching a leaf of the tree).

- The greedy search is guaranteed to reach a leaf because of an optimisation strategy: we will store **an entire vector of modifications** if it leads to a utility passing some predetermined threshold. This threshold can be modified on line ```195```.

- The output is stored in ```data/mcts_result_X```, where X is the number of modifications.

- Usage: ```mcts.py```

6. ```taxienv.py```

- This file contains the implementation of the ```Taxi-v3``` environment, based on OpenAI Gym. Here the code is modified to allow modifications to the environment.

- There are two kinds of modifications to an environment. For an environment ```env``` of class ```TaxiEnv```, its walls can be found by outputting ```env.walls```. To find the cells that allow diagonal moves, we simply output ```env.special```. The modifications allowed are to cut down walls, or to include more special cells.

- To cut down a wall (walls), we initialize a new environment ```new_env = env.transition(A)```, where ```A``` is a list of walls to be cut down. Walls are represented as tuples on the string representation of the environment, whereas cells are represented as tuples on the 5x5 square representation (so ```(0, 0)``` means top left square).

- To introduce a new special cell, we append it to ```env.special``` via ```env.special.append(c)```, where ```c``` is a tuple representing the cell. 

- Example, to take down wall ```(1, 4)``` (here ```(1, 4)``` is the position of the wall in the **string** representation of the environment), and add cell ```(2, 2)``` to that environment, these lines of code can be written in order:

```new_env = env.transition([(1, 4)])```

```new_env.special.append((2, 2))```

- To take down two walls, ```(1, 4)``` and ```(5, 2)``` from some environment ```env```, granted they have not been cut off yet, we write

```new_env = env.transition([(1, 4), (5, 2)])```

- To visualize the environment ```env```, we write ```env.render()``` 

- This file is used as a helper file.
