"""
Run RL experiments across multiple algorithms, environments, and seeds.

Usage:
    python run_experiment.py                          # Run all experiments
    python run_experiment.py --algo PPO --env empty   # Run single config
    python run_experiment.py --plot                   # Plot saved results
"""

import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# CONFIGURATION - Edit these to change what experiments to run
# ============================================================================

ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}

ENVIRONMENTS = {
    "empty_5x5": "MiniGrid-Empty-5x5-v0",
    "empty_8x8": "MiniGrid-Empty-8x8-v0",
    "empty_16x16": "MiniGrid-Empty-16x16-v0",
    "doorkey_5x5": "MiniGrid-DoorKey-5x5-v0",
    "doorkey_8x8": "MiniGrid-DoorKey-8x8-v0",
}

SEEDS = [1, 2, 3, 4, 5]
TOTAL_TIMESTEPS = 50000
RESULTS_DIR = "results"

# ============================================================================
# CUSTOM CNN FOR MINIGRID
# ============================================================================


class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 512
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]  # (H, W, C)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float() / 255.0
            sample = sample.permute(0, 3, 1, 2)  # (batch, C, H, W)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float() / 255.0
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


# ============================================================================
# CALLBACK - Records metrics during training
# ============================================================================


class MetricsCallback(BaseCallback):
    """Records episode rewards and lengths during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self._last_num_episodes = 0

    def _on_step(self) -> bool:
        # Only record when a NEW episode finishes
        num_episodes = len(self.model.ep_info_buffer)
        if num_episodes > self._last_num_episodes:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info["r"])
            self.episode_lengths.append(ep_info["l"])
            self.timesteps.append(self.num_timesteps)
            self._last_num_episodes = num_episodes
        return True


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================


def make_env(env_name):
    """Create and wrap a MiniGrid environment."""
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env


def run_single_experiment(algo_name, env_key, seed):
    """Run a single experiment and return metrics."""

    print(f"\n{'=' * 60}")
    print(f"Running: {algo_name} on {env_key} (seed={seed})")
    print(f"{'=' * 60}")

    # Setup
    algo_class = ALGORITHMS[algo_name]
    env_name = ENVIRONMENTS[env_key]
    env = make_env(env_name)
    log_dir = f"runs/{algo_name}_{env_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create model
    model = algo_class(
        "CnnPolicy",
        env,
        verbose=0,
        seed=seed,
        device="cuda",
        tensorboard_log=log_dir,
    )
    # model = algo_class(
    #     "CnnPolicy",
    #     env,
    #     verbose=0,
    #     seed=seed,
    #     device="cuda",
    #     policy_kwargs={"features_extractor_class": MiniGridCNN},
    #     tensorboard_log=log_dir,
    # )

    # Train with callback
    callback = MetricsCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # Final evaluation (10 episodes)
    eval_rewards = []
    for episode in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        eval_rewards.append(total_reward)

    env.close()

    results = {
        "algo": algo_name,
        "env": env_key,
        "seed": seed,
        "timesteps": callback.timesteps,
        "episode_rewards": callback.episode_rewards,
        "episode_lengths": callback.episode_lengths,
        "final_eval_mean": float(np.mean(eval_rewards)),
        "final_eval_std": float(np.std(eval_rewards)),
    }

    print(
        f"Final eval: {results['final_eval_mean']:.3f} ± {results['final_eval_std']:.3f}"
    )

    return results


def save_results(results, filename):
    """Save results to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {filepath}")


def load_all_results():
    """Load all results from the results directory."""
    all_results = []
    if not os.path.exists(RESULTS_DIR):
        return all_results
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, filename)) as f:
                all_results.append(json.load(f))
    return all_results


def run_all_experiments():
    """Run all algorithm/environment/seed combinations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for algo_name in ALGORITHMS:
        for env_key in ENVIRONMENTS:
            for seed in SEEDS:
                results = run_single_experiment(algo_name, env_key, seed)
                filename = f"{algo_name}_{env_key}_seed{seed}_{timestamp}.json"
                save_results(results, filename)


# ============================================================================
# PLOTTING
# ============================================================================


def plot_results():
    """Generate comparison plots from saved results."""
    all_results = load_all_results()

    if not all_results:
        print("No results found in results/ directory")
        return

    # Group by environment
    envs = set(r["env"] for r in all_results)
    algos = set(r["algo"] for r in all_results)

    os.makedirs("plots", exist_ok=True)

    for env_key in envs:
        plt.figure(figsize=(10, 6))

        for algo_name in algos:
            # Get all runs for this algo/env
            runs = [
                r for r in all_results if r["algo"] == algo_name and r["env"] == env_key
            ]

            if not runs:
                continue

            # Average across seeds (simple version - just plot final eval)
            means = [r["final_eval_mean"] for r in runs]

            print(
                f"{env_key} | {algo_name}: {np.mean(means):.3f} ± {np.std(means):.3f}"
            )

        # Bar plot of final performance
        algo_names = list(algos)
        means = []
        stds = []

        for algo_name in algo_names:
            runs = [
                r for r in all_results if r["algo"] == algo_name and r["env"] == env_key
            ]
            if runs:
                vals = [r["final_eval_mean"] for r in runs]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(algo_names))
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        plt.xticks(x, algo_names)
        plt.ylabel("Final Evaluation Reward")
        plt.title(f"Algorithm Comparison: {env_key}")
        plt.tight_layout()

        plot_path = f"plots/{env_key}_comparison.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiments")
    parser.add_argument(
        "--algo", type=str, choices=list(ALGORITHMS.keys()), help="Algorithm to use"
    )
    parser.add_argument(
        "--env", type=str, choices=list(ENVIRONMENTS.keys()), help="Environment to use"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot results instead of running experiments",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all algorithm/environment/seed combinations",
    )

    args = parser.parse_args()

    if args.plot:
        plot_results()
    elif args.all:
        run_all_experiments()
    elif args.algo and args.env:
        results = run_single_experiment(args.algo, args.env, args.seed)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.algo}_{args.env}_seed{args.seed}_{timestamp}.json"
        save_results(results, filename)
    else:
        print(__doc__)
        print("\nExamples:")
        print("  python run_experiment.py --algo PPO --env empty_5x5 --seed 42")
        print("  python run_experiment.py --all")
        print("  python run_experiment.py --plot")
