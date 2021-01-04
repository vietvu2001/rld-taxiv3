This file contains the protocols for conducting the experiments regarding each presented technique.

1. Supervised Learning: ```allstates.py```, ```allstates_trimmed.py```, ```connected_allstates.py```, ```connected_allstates_trimmed.py```

Number of modifications and corresponding number of iterations:

- 1: 50 iterations
- 2: 200 iterations
- 3: 1500 iterations
- 4: 6000 iterations
- 5: 7500 iterations
- 6: 8000 iterations

Number of trials conducted: 5

- Note: Here we conduct data collection once, but feed it to the ANN 5 times.

2. MCTS: ```mcts.py```, ```mcts_trimmed.py```, ```connect_mcts.py```, ```connect_mcts_trimmed.py```

Number of modifications and corresponding number of modifications:

- 1: 50 iterations
- 2: 200 iterations
- 3: 1000 iterations
- 4: 1500 iterations
- 5: 2500 iterations
- 6: 3000 iterations

Number of trials conducted: 5

3. Batch Greedy (dissecting the MCTS tree): ```batch_greedy.py```

Number of modifications and corresponding combinations:

- 1: ```1```
- 2: ```1 + 1```, ```2```
- 3: ```1 + 1 + 1```, ```2 + 1```, ```3```
- 4: ```1 + 1 + 1 + 1```, ```2 + 2```, ```3 + 1```, ```4```
- 5: ```1 + 1 + 1 + 1 + 1```, ```2 + 2 + 1```, ```3 + 2```, ```4 + 1```
- 6: ```1 + 1 + 1 + 1 + 1 + 1```, ```2 + 2 + 2```, ```3 + 3```, ```4 + 2```