This markdown contains the aims and instructions for running each Python file in ```src/```.

1. ```allstates.py```

- This file collects the training data needed for Supervised Learning. We create environments using random vectors of modifications (fixed number of iterations), and evaluate them using Q-Learning agents. To specify how many modifications are allowed, modify line ```95```, along with line ```85``` to figure out how many rounds you want to do (based on the number of environments you want to generate).

- The data we store consists of the vector of modifications, and the utility of a Q-Learning agent in that environment. Here utility is defined as the average taken over the total rewards from all possible starting positions. This data is stored in ```data/data_X.csv``` (where X is the number of modifications).

- Usage: ```python allstates.py```

2. ```feed.py```

- This file reads the data in ```data/data_X.csv``` and performs training of a model, in which the input is the random vector of modifications and the output is the utility. The random vector of modifications is flattened. 
- We use this model to predict the utility of a Q-Learning agent in a large number of environments. You can tune this number on lines ```273``` and ```276```. We build a minimum heap to store a small number of environments with highest predictions, and do Q-Learning evaluations on them to get exact numbers for comparison.
- The output contains the chosen vector of modifications, along with the corresponding utility. It is stored in ```data/sl_result_X.txt``` (where X is the number of modifications).

- Usage: ```python feed.py```

3. ```heuristic.py```

- This file contains the implementations of the ```cell_frequency``` and ```wall_interference``` heuristics. The input is an agent of class ```Agent``` (as stored in ```qlearn.py```), in which its environment is a field. We return the rankings of the cells that the agent crosses the most, the the rankings of the walls for whose elimination will improve the rewards from several starting states (rank by how many starting states can be improved). 

- We also implement the ```utility``` function, which also takes an agent of class ```Agent``` as input, and returns the utility of the agent in its respective environment.

- Usage: ```python heuristic.py```

4. ```qlearn.py```

- This file implements the training and evaluation of a Q-Learning agent in its respective environment. The class ```Agent``` is implemented in this file.

- Usage: ```qlearn.py```

5. ```mcts.py```

- This file implements the Monte Carlo Tree Search algorithm for the problem. The number of modifications corresponds to the maximum layer of the tree, and denoted as ```max_layer``` in the file. Thus, to modify the number of modifications, modify the value of ```max_layer``` on line ```32```. The number of iterations on which the tree is explored can be modified on line ```415```.

- A greedy search through the 
