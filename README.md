# minigrid-rl-experiments

Reinforcement learning experiments using MiniGrid environments, developed as part of the Statistical Planning and Reinforcement Learning module.

As of now, I've just added some super simple code that renderes the 5x5 grid. Just run the code below in the terminal if you haven't installed any of the dependencies already and click run.


## Setup
```bash
pip install gymnasium minigrid numpy torch
```


## Project Overview

This project has two parts:

### Part 1: Frozen Lake
Implementation of core RL algorithms on the Frozen Lake environment:
- Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration
- Tabular Sarsa and Q-Learning
- Linear function approximation (Sarsa and Q-Learning)
- Deep Q-Network (DQN)

### Part 2: Beyond Frozen Lake
Custom experiments using MiniGrid environments to explore RL challenges such as:
- Sample efficiency
- Partial observability
- Sparse rewards
- Exploration strategies



- [MiniGrid Documentation](https://minigrid.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
