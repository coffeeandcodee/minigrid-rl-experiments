# DoorKey-8x8 Exploration Experiments

All methods evaluated on **clean environment** (no intrinsic rewards during evaluation).

| Method | Timesteps | Seeds | Mean Reward | Success Rate |
|--------|-----------|-------|-------------|--------------|
| PPO Baseline | 200k | 5 | 0.000 | 0/5 |
| PPO + Count-based | 200k | 5 | 0.000* | 0/5* |
| PPO + ICM | 200k | 3 | 0.000 | 0/3 |
| PPO + RND | 200k | 3 | 0.000 | 0/3 |
| PPO + ICM | 500k | 3 | 0.000 | 0/3 |
| **PPO + Reward Shaping** | **500k** | 3 | **0.549** | **2/3** |

*Count-based originally reported high rewards (4.5+) because evaluation included intrinsic bonuses. When evaluated on clean environment, success rate was 0%.

## Key Finding
At equal compute (500k timesteps), reward shaping achieved 67% success while ICM achieved 0%.
