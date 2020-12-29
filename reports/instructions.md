This markdown contains the aims and instructions for running each Python file in ```src/```.

1. ```allstates.py```

- This file collects the training data needed for Supervised Learning. We create environments using random vectors of modifications (fixed number of iterations), and evaluate them using Q-Learning agents. To specify how many modifications are allowed, modify line ```95```, along with line ```85``` to figure out how many rounds you want to do (based on the number of environments you want to generate).

- The data we store consists of the vector of modifications, and the utility of a Q-Learning agent in that environment. Here utility is defined as the average taken over the total rewards from all possible starting positions. This data is stored in ```data/data_[X].csv``` (where $X$ is the number of modifications).

2. ```feed.py```

- This file 