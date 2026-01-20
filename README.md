# Deep Reinforcement Learning Experiments

Experimental investigation of key challenges in deep reinforcement learning using MiniGrid environments and Stable Baselines 3.

## Challenges Investigated

| Challenge | Environment | Key Finding |
|-----------|-------------|-------------|
| **Training Instability** | Empty-5x5, Empty-8x8 | DQN learns then catastrophically collapses; policy gradient methods (PPO, A2C) remain stable |
| **Sparse Rewards** | DoorKey-5x5, DoorKey-8x8 | Multi-step prerequisites dramatically reduce sample efficiency; reward shaping enables learning |
| **Generalization** | DistShift1, DistShift2 | Policies memorize fixed action sequences rather than learning reactive behavior |

## Installation

```bash
git clone <repo-url>
cd DRL_v2
pip install gymnasium minigrid stable-baselines3 sb3-contrib matplotlib numpy
```

## Reproducing Experiments

### Basic Usage

```bash
# Run a single experiment
python3 run_experiment.py --algo PPO --env empty_5x5 --seed 1

# Run all algorithms on an environment (5 seeds each)
python3 run_experiment.py --env empty_5x5

# Custom timesteps
python3 run_experiment.py --algo PPO --env doorkey_5x5 --timesteps 150000
```

### Key Experiments

```bash
# Training instability (DQN collapse)
python3 run_experiment.py --env empty_5x5

# Sparse rewards
python3 run_experiment.py --env doorkey_5x5 --timesteps 150000
python3 run_experiment.py --algo PPO --env doorkey_8x8 --timesteps 200000

# Generalization
python3 run_experiment.py --generalization
```

### Generate Visualizations

```bash
# Learning curves for an environment
python3 run_experiment.py --env empty_5x5 --plot

# Combined summary across all environments
python3 run_experiment.py --combined
python3 run_experiment.py --summary
```

## Project Structure

```
DRL_v2/
├── run_experiment.py      # Main experiment runner
├── results/               # JSON files with experiment data
│   ├── empty_5x5/
│   ├── doorkey_5x5/
│   ├── doorkey_8x8/
│   └── generalization/
├── plots/                 # Generated visualizations
│   ├── empty_5x5/
│   ├── doorkey_5x5/
│   ├── doorkey_8x8/
│   └── combined/
└── resources/             # Reference materials
```

## Algorithms

- **PPO** (Proximal Policy Optimization)
- **A2C** (Advantage Actor-Critic)
- **DQN** (Deep Q-Network)
- **QRDQN** (Quantile Regression DQN)

## Environments

| Environment | Description |
|-------------|-------------|
| `empty_5x5` | Simple 5×5 navigation |
| `empty_8x8` | Larger 8×8 navigation |
| `doorkey_5x5` | Must find key, unlock door, reach goal |
| `doorkey_8x8` | Larger version with sparser rewards |
| `distshift1/2` | Generalization test environments |

## References

- Arulkumaran et al. (2017). A Brief Survey of Deep Reinforcement Learning. *IEEE Signal Processing Magazine*.
- Cobbe et al. (2019). Quantifying Generalization in Reinforcement Learning. *ICML*.
- Chevalier-Boisvert et al. (2023). Minigrid & Miniworld. *NeurIPS*.
