# MiniGrid RL Experiments

Assignment Part 2: Beyond the Frozen Lake

Here's some ideas as to what algorithms and experiments we can look into testing in the days to come. 

A lot of the algorithms are available in Stables Baselines 3 (the library mentioned in the assignment.pdf).



---

## Setup

```bash
pip install gymnasium minigrid stable-baselines3 sb3-contrib numpy torch matplotlib
```

---

## Algorithms We Can Use

All available in **Stable Baselines 3** (no custom implementation needed):

| Algorithm | What it is |
|-----------|------------|
| **DQN** | Value-based, discrete actions |
| **A2C** | Basic actor-critic |
| **PPO** | Improved actor-critic (more stable than A2C) |
| **SAC** | Off-policy actor-critic |

**Exploration bonus** (from SB3-Contrib):
- **PPO + ICM** — adds curiosity-driven exploration

---

## Challenges We Can Address

From the Arulkumaran 2017 paper (referenced in assignment):

1. **Sample Efficiency** — How quickly does the agent learn?
2. **Sparse Rewards** — Can the agent learn when rewards are rare?
3. **Exploration** — Does curiosity help discover solutions faster?
4. **Generalization** — Can agents trained on small envs work on larger ones?

---

## Environments We Can Use

**MiniGrid** — recommended in the assignment, easy to install, many variants.

| Environment | What's hard about it |
|-------------|---------------------|
| `MiniGrid-Empty-8x8-v0` | Nothing (sanity check) |
| `MiniGrid-DoorKey-5x5-v0` | Must find key, unlock door, reach goal |
| `MiniGrid-DoorKey-16x16-v0` | Same but larger (harder exploration) |
| `MiniGrid-MultiRoom-N2-S4-v0` | Navigate through 2 connected rooms |
| `MiniGrid-MultiRoom-N6-v0` | Navigate through 6 rooms (very sparse reward) |

---

## Experiments We Can Run

### Experiment 1: Algorithm Comparison
> Which algorithm learns fastest on DoorKey?

Compare A2C vs PPO vs DQN on the same environment. Plot learning curves.

### Experiment 2: Exploration Bonus
> Does curiosity help with sparse rewards?

Compare PPO vs PPO+ICM on MultiRoom environments. Measure success rate.

### Experiment 3: Generalization
> Can a small-env agent work on bigger envs?

Train on DoorKey-5x5, test on DoorKey-8x8 and 16x16.

---

## Metrics to Record

- Episode reward over training steps
- Success rate (did agent reach goal?)
- Time to first success
- Wall-clock training time

Run each experiment with **5 different random seeds** to get error bars.

---

## Resources

- [MiniGrid Docs](https://minigrid.farama.org/)
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [SB3-Contrib](https://sb3-contrib.readthedocs.io/)
