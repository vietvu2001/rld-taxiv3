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

- More details are explained in the documentation of ```greedy.py```.

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

- To cut down a wall (walls), we initialize a new environment ```new_env = env.transition(A)```, where ```A``` is a list of walls to be cut down. Walls are represented as tuples on the string representation of the environment, whereas cells are represented as tuples on the 5x5 square representation (so ```(0, 0)``` means top left square). The string representation of the environment is on lines ```14-22```.

- To introduce a new special cell, we append it to ```env.special``` via ```env.special.append(c)```, where ```c``` is a tuple representing the cell. 

- Example, to take down wall ```(1, 4)``` (here ```(1, 4)``` is the position of the wall in the **string** representation of the environment), and add cell ```(2, 2)``` to that environment, these lines of code can be written in order:

```new_env = env.transition([(1, 4)])```

```new_env.special.append((2, 2))```

- To take down two walls, ```(1, 4)``` and ```(5, 2)``` from some environment ```env```, granted they have not been cut off yet, we write

```new_env = env.transition([(1, 4), (5, 2)])```

- To visualize the environment ```env```, we write ```env.render()``` 

- This file is used as a helper file.

7. ```greedy.py```

- This file implements the greedy algorithm. This algorithm is basically as follows: we start with an original environment ```env```. We make an agent of class ```Agent``` and train it:

```agent = QAgent(env)```

```agent.qlearn(600)```

- Then, based on ```heuristic.py```, we can run

```cell_frequency(agent)```

```wall_interference(agent)```

to get the rankings mentioned in ```heuristic.py```. The outputs are dictionaries.

- From here, we get the max value of each dictionary, and find **all** tuples of walls and cells that have this max value in their respective dictionaries (walls have their own dictionary from ```wall_interference```, and cells have their own dictionary from ```cell_frequency```). We make the environments from these tuples and train on them to find out which tuple (between wall and cell) that leads to the highest utility, from the function ```utility``` that takes as input an agent of class ```Agent```. From here, we make the modification based on whether this found tuple is a wall or cell.

- Repeat this process until we have reached the budget (the number of repetitions is equal to the budget).

- Return the found vector of modifications and corresponding utility.

- The output is stored in ```data/greedy_X.txt```, where X is the number of modifications.

- Usage: ```python greedy.py```

8. ```multi_greedy.py```

- Same as ```greedy.py``, but here we use parallel computing to reduce the time needed to train.

9. **Please read: An introduction to reductions of possible modifications**

- This section does not contain any code, but an idea. To improve the efficiency of Supervised Learning and Monte Carlo Tree Search methods, we can reduce the number of modifications. This stems from the observation that, for example, we do not really need to add diagonal moves to cell ```(4, 0)``` on the 5x5 representation of environment, because it is not possible to move diagonally as there are two walls blocking it anyway!

- We use the ```cell_frequency``` heuristic in ```heuristic.py``` to chop down the number of possible modifications to only **those that matter**. Specifically, we reduce the number of modifications from 31 to 20, by picking all the walls (there are 6 walls), and 14 cells that have the highest rankings out of 25 cells by output of ```cell_frequency``` on the original environment. By reducing nearly half the number of modifications, we drastically reduce the total number of possible modifications.

- On the MCTS tree, deeper searches can be made. On Supervised Learning, better efficiency in learning vectors of modifications and outputs can be achieved. **Theoretically**.

10. ```allstates_trimmed.py```

- Please read point 9. 

- This is the same implementation of ```allstates.py```, but the main difference here is that the list ```modifications``` has been modified from all modifications to only 20 (this can be seen on lines ```95 - 105```).

- The number of modifications can be modified on line ```90```, and the number of rounds on line ```80```.

- The output will be stored to ```data/data_trimmed_X.csv```, where X is the number of modifications.

- Usage: ```python allstates_trimmed.py```

11. ```feed_trimmed.py```

- Please read point 9.

- This is the same implementation of ```feed.py```, but the main difference here is that it only takes input from ```data/data_trimmed_X.csv```.

- The number of modifications can be modified on line ```42```. The output is stored in ```data/sl_trimmed_result_X.txt```, where X is the number of modifications.

- Usage: ```python feed_trimmed.py```

12. ```mcts_trimmed.py```

- Please read point 9. 

- This is the same implementation of ```mcts.py```, but the main difference here is that the list of modifications has been changed. This change can be seen on lines ```145 - 154```.

- The output is stored in ```data/mcts_trimmed_result_X.txt```, where X is the number of modifications.

- Usage: ```python mcts_trimmed.py```

13. **Please read: An introduction to connected Q-Learning training**

- This section does not contain code, but another idea. The main purpose of connected training is to improve on the time needed to train. As the reader may already find out in point 7, the number of iterations needed to train a Q-Learning agent is ```600```. We can reduce this number by **cloning a Q-Learning agent already trained in the original environment** to train in the modified environment. This will reduce the number of iterations, but **it is unfortunately only an approximation**. 

- The error is often found to be 1 to 4 over 300 starting states, taking the sum of rewards. However, since we are taken the mean of the rewards, the error is divided by 300, which is an error we are willing to accept for time decrease.

- In ```greedy.py```, we can even use an agent used in the previous iteration to train in the new iteration. Since the environments in this algorithm differ by only one modification from one iteration to the next, the connected training becomes much more robust (compared to a modified environment with, say, 5 modifications and an agent from the original environment). This idea has **already been implemented**.

14. ```connect_qlearn.py```

- Please read point 13.

- This file implements the function for **connected Q-Learning**, based on the idea discussed in point 13.

- This file works as a helper file.

15. ```connect_potency.py```

- Please read point 13.

- This file experiments with the number of iterations we can use to train a modified environment, based on the idea of **connected training**.

- The idea is simple: let the number of iterations be ```N```. Then we compare a connected agent trained in ```N``` iterations to a normal agent trained in 600 iterations. The error is defined as the difference in the sum of rewards, taken over all starting states.

- We use parallel computing and make a list of error values, and try to minimize the average error as well as keep ```N``` far away enough from 600. After experimenting, we find out that ```N = 400``` is a reasonable choice. Crudely, 33% of training time can be saved.

- The value of ```N``` can be modified on line ```87```, where ```400``` is set as default.

- This file is only an experiment, but is useful for solving the problem.

- Usage: ```python connect_potency.py```

16. ```connected_allstates.py```

- Please read point 13.

- This file basically implements the same idea in ```allstates.py```, but the main difference is that the agents used to train in random environments and collect utilities are all connected agents. This change can be seen on lines ```83 - 85```, where the function ```connected_qlearn_as_func``` is implemented, and below ```main``` on line ```114```.

- The output is stored in ```data/connected_data_X.csv```, where X is the number of modifications.

- Usage: ```python connected_allstates.py```

17. ```connected_feed.py```

- Please read point 13.

- This file takes input from ```data/connected_data_X.csv``` and implements the same idea as ```feed.py``` (the main difference is that we are using connected training to gather data). The number of modifications, ```num_mods```, can be modified on line ```41```. 

- The output is stored in ```data/connected_sl_result_X.txt```, where X is the number of modifications.

- Usage: ```python connected_feed.py```

18. ```connect_mcts.py```

- Please read point 13. 

- This file implements the same idea as ```mcts.py```, but the main difference here is that the agents used for playout phases are connected agents from a trained agent in the original environment located at the root. The number of modifications, or ```max_layer```, can be modified on line ```33```.

- The output is stored in ```data/connect_mcts_result_X.txt``` where X is the number of modifications.

- Usage: ```python connect_mcts.py```

19. ```connected_allstates_trimmed.py```

- Please read points 9 and 13.

- This file implements the same idea as ```allstates_trimmed.py```, but the main difference is that agents are connected to a trained agent in the original environment. The values of ```rounds``` and ```num_mods``` can be modified in lines ```79``` and ```89```.

- The output is stored in ```data/connected_data_trimmed_X.csv```, where X is the number of modifications.

- Usage: ```python connected_allstates_trimmed.py```

20. ```connected_feed_trimmed.py```

- Please read points 9 and 13. 

- This file implements the same idea as ```feed_trimmed.py```, but it only takes in data from ```data/connected_data_trimmed_X.csv``` (the difference is that we are using connected training to gather data). The number of modifications can be modified on line ```42```.

- The output is stored in ```data/connected_sl_trimmed_result_X.txt```, where X is the number of modifications.

- Usage: ```python connected_feed_trimmed.py```

21. ```connect_mcts_trimmed.py```

- Please read points 9 and 13.

- This file implements the same idea as ```mcts_trimmed.py```, but the agents used for playout phases are connected to a trained agent of the original environment located at the root. The number of modifications, or ```max_layer```, can be modified on line ```34```.

- The output is stored in ```data/connect_mcts_trimmed_result_X.txt```, where X is the number of modifications.

- Usage: ```python connect_mcts_trimmed.py```

22. ```correct_data.py```

- This file is used for multiprocessing debugging when we implemented ```allstates.py```.

23. ```path_based.py```

- This file is a deprecated version of a greedy solution to this problem.

