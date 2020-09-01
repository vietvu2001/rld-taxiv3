# Project

This repository contains the code for my project, Three Methods for Reinforcement Learning Design.

# Packages to install
- gym: provides text-based representation render through utils
- tensorflow: training ANN for the Supervised Learning method
- termcolor
- sklearn: provides the function train_test_split for separating training - testing data

# Usage
1. Supervised Learning
- First, run allstates.py to store sampled data to data_3.csv
- Then, run feed.py to train the ANN and return the result of this method in sl_result_4 (budget is 4).
- Dynamic version (allowing dynamic budget changes) will be updated shortly.

2. Monte Carlo Tree Search (MCTS)
- Run mcts.py
- The tree's contents will be stored into tree_2.csv, and the result into mcts_result.txt
- Default budget is 4. To change the budget size to N, replace 4 in line 30 (of mcts.py) to N.

3. Path-based Method
- Run path_based.py
- The result will be stored in path_based_result.txt

4. MCTS trimmed by Path-based Heuristics
- Run mcts_trimmed.py
- The tree's contents will be stored into tree.csv, and the result into mcts_trimmed_result.txt
- Default budget is 4. To change the budget size to N, replace 4 in line 30 (of mcts_trimmed.py) to N.
