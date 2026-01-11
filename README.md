# MiniGrid RL Experiments

Reinforcement learning experiments using MiniGrid environments, developed as part of the **Statistical Planning and Reinforcement Learning** module (Assignment Part 2: Beyond the Frozen Lake).

---

## Table of Contents

1. [Setup](#setup)
2. [Project Goal](#project-goal)
3. [Recommended Algorithms](#recommended-algorithms)
4. [Recommended Environments](#recommended-environments)
5. [Experiment Design](#experiment-design)
6. [Our Proposed Study](#our-proposed-study)
7. [Resources](#resources)

---

## Setup

```bash
pip install gymnasium minigrid stable-baselines3 sb3-contrib numpy torch matplotlib
```

---

## Project Goal

Based on the assignment requirements, we need to:

- Use **existing libraries** (Minigrid, Stable Baselines 3) — don't reinvent the wheel
- Address **recognized RL challenges** from Arulkumaran et al., 2017 (Section VI)
- Design **creative/challenging experiments** (worth 35/50 points!)
- Create a **reproducible experimental protocol** (worth 10/50 points)
- Conduct **rigorous analysis** with tables and plots (worth 5/50 points)

---

## Recommended Algorithms

We have three possible directions. Each option trades off between safety/documentation and creativity/risk.

### Option A: Sample Efficiency Comparison ✅ (Safest, well-documented)

| Algorithm | Library | Why |
|-----------|---------|-----|
| **PPO** (Proximal Policy Optimization) | Stable Baselines 3 | Strong baseline, very stable |
| **SAC** (Soft Actor-Critic) | Stable Baselines 3 | Sample-efficient for continuous action spaces |
| **DQN** (with variants: Double, Dueling, PER) | Stable Baselines 3 | Compare improvements over vanilla DQN |

### Option B: Exploration Challenge 🔍 (More creative)

| Algorithm | Library | Why |
|-----------|---------|-----|
| **PPO + ICM** (Intrinsic Curiosity Module) | SB3-Contrib | Addresses sparse rewards via curiosity |
| **RND** (Random Network Distillation) | Custom / TorchRL | State-of-the-art exploration bonus |
| **PPO** (baseline) | Stable Baselines 3 | Control comparison |

### Option C: Hierarchical RL 🏗️ (Challenging, high reward potential)

| Algorithm | Library | Why |
|-----------|---------|-----|
| **HIRO** or **Option-Critic** | TorchRL / Custom | Multi-level temporal abstraction |
| **PPO** | Stable Baselines 3 | Flat baseline for comparison |

---

## Recommended Environments

**MiniGrid** is explicitly mentioned in the assignment as a good choice. It's perfect because:

- ✅ Highly customizable
- ✅ Partial observability (a recognized RL challenge!)
- ✅ Sparse rewards (a recognized RL challenge!)
- ✅ Easy to create a curriculum of increasing difficulty
- ✅ Well-maintained by the Farama Foundation

### Specific Environments to Consider

| Environment | Challenge Type | Difficulty |
|-------------|----------------|------------|
| `MiniGrid-Empty-8x8-v0` | Basic navigation | Easy (sanity check) |
| `MiniGrid-DoorKey-5x5-v0` | Object manipulation, sparse reward | Medium |
| `MiniGrid-DoorKey-16x16-v0` | Object manipulation, sparse reward | Hard |
| `MiniGrid-KeyCorridorS3R3-v0` | Multi-step planning, sparse reward | Hard |
| `MiniGrid-MultiRoom-N2-S4-v0` | Exploration, sparse reward | Medium |
| `MiniGrid-MultiRoom-N6-v0` | Exploration, very sparse reward | Very Hard |
| `MiniGrid-MemoryS17Random-v0` | Memory requirement | Hard |

### Environment Notes

- **DoorKey**: Agent must pick up a key to unlock a door and reach the goal
- **MultiRoom**: Agent navigates through multiple connected rooms to find the goal
- **KeyCorridor**: Agent must navigate corridors and use keys strategically

---

## Experiment Design

### Experiment 1: Sample Efficiency Study

**Question**: How many environment steps do different algorithms need to solve tasks of increasing difficulty?

| Component | Details |
|-----------|---------|
| **Independent Variable** | Algorithm (PPO vs SAC vs DQN variants) |
| **Dependent Variable** | Cumulative reward over training steps |
| **Control** | Same environment, same random seeds |
| **Replication** | 5-10 runs per condition |

### Experiment 2: Sparse Reward Robustness

**Question**: Do exploration bonuses (intrinsic curiosity) help in sparse-reward environments?

| Component | Details |
|-----------|---------|
| **Independent Variable** | Exploration method (None vs ICM vs RND) |
| **Dependent Variable** | Success rate, time to first success |
| **Environment** | MiniGrid-MultiRoom (increasing number of rooms) |
| **Replication** | 5-10 runs per condition |

### Experiment 3: Generalization / Transfer

**Question**: Can agents trained on small environments generalize to larger versions?

| Component | Details |
|-----------|---------|
| **Training Environment** | MiniGrid-DoorKey-5x5 |
| **Test Environments** | MiniGrid-DoorKey-8x8, 16x16 |
| **Metrics** | Zero-shot performance, fine-tuning speed |

---

## Our Proposed Study

### Title
**"Sample Efficiency and Exploration in Sparse-Reward Grid Worlds"**

### Research Question
> Does intrinsic curiosity improve sample efficiency in sparse-reward navigation tasks?

### Design Summary

| Component | Choice |
|-----------|--------|
| **Environments** | MiniGrid DoorKey and MultiRoom series (varying sizes) |
| **Algorithms** | PPO, PPO+ICM, DQN+PER |
| **Libraries** | Stable Baselines 3, SB3-Contrib, MiniGrid |
| **Metrics** | Episode return vs. training steps, success rate, wall-clock time |
| **Protocol** | 5 seeds per condition, learning curves with 95% confidence intervals |

### Why This Works

- ✅ **Creative enough** to score well on the 35-point creativity component
- ✅ **Uses established libraries** (SB3, MiniGrid) — no time wasted on implementation bugs
- ✅ **Clear hypothesis** that can be tested and analyzed rigorously
- ✅ **Reproducible** with fixed seeds and documented hyperparameters
- ✅ **Addresses a recognized RL challenge** (exploration in sparse rewards)

---

## Resources

- [MiniGrid Documentation](https://minigrid.farama.org/)
- [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/)
- [SB3-Contrib (includes ICM, RND)](https://sb3-contrib.readthedocs.io/)
- [Arulkumaran et al., 2017 — A Brief Survey of Deep RL](https://arxiv.org/abs/1708.05866)
- [Farama Foundation](https://farama.org/)

---

## Team Discussion Points

1. **Which option (A, B, or C) should we pursue?**
2. **Which specific MiniGrid environments should we use?**
3. **How should we divide the work?**
   - Environment setup / wrappers
   - Training scripts
   - Evaluation / plotting
   - Report writing

---

*Last updated: January 2026*
