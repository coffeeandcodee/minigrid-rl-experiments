"""
Run RL experiments across multiple algorithms, environments, and seeds.

Usage:
    python run_experiment.py --algo PPO --env empty_5x5   # Run single config
    python run_experiment.py --all                        # Run all experiments
    python run_experiment.py --table                      # Print results table
    python run_experiment.py --plot                       # Plot saved results
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
from datetime import datetime

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
    "doorkey_5x5": "MiniGrid-DoorKey-5x5-v0",
    "doorkey_8x8": "MiniGrid-DoorKey-8x8-v0",
}

SEEDS = [1, 2, 3, 4, 5] 
TOTAL_TIMESTEPS = 50_000
RESULTS_DIR = "results"

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
        
    def _on_step(self) -> bool:
        # Check infos for episode completion (works even when buffer is full)
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps.append(self.num_timesteps)
        return True

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def make_env(env_name):
    """Create and wrap a MiniGrid environment."""
    env = gym.make(env_name)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env


def run_single_experiment(algo_name, env_key, seed):
    """Run a single experiment and return metrics."""
    
    print(f"\n{'='*60}")
    print(f"Running: {algo_name} on {env_key} (seed={seed})")
    print(f"{'='*60}")
    
    # Setup
    env_name = ENVIRONMENTS[env_key]
    env = make_env(env_name)
    algo_class = ALGORITHMS[algo_name]
    
    # Create model
    model = algo_class("MlpPolicy", env, verbose=0, seed=seed)
    
    # Train with callback
    callback = MetricsCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    
    # Final evaluation (10 episodes)
    eval_rewards = []
    for _ in range(10):
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
    
    print(f"Final eval: {results['final_eval_mean']:.3f} ± {results['final_eval_std']:.3f}")
    
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
# TABLES
# ============================================================================

def print_table():
    """Print a formatted table of all results."""
    all_results = load_all_results()
    
    if not all_results:
        print("No results found in results/ directory")
        return
    
    # Get unique algos and envs
    algos = sorted(set(r["algo"] for r in all_results))
    envs = sorted(set(r["env"] for r in all_results))
    
    # Header
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Table 1: Mean ± Std by Algorithm and Environment
    print("\n### Final Evaluation Reward (mean ± std across seeds)\n")
    
    col_width = 18
    
    # Header row
    header = "Algorithm".ljust(12) + "".join(e.ljust(col_width) for e in envs)
    print(header)
    print("-" * len(header))
    
    # Data rows
    for algo in algos:
        row = algo.ljust(12)
        for env in envs:
            runs = [r for r in all_results if r["algo"] == algo and r["env"] == env]
            if runs:
                vals = [r["final_eval_mean"] for r in runs]
                mean, std = np.mean(vals), np.std(vals)
                row += f"{mean:.3f} ± {std:.3f}".ljust(col_width)
            else:
                row += "-".ljust(col_width)
        print(row)
    
    # Table 2: Individual runs
    print("\n\n### All Individual Runs\n")
    print(f"{'Algorithm':<10} {'Environment':<15} {'Seed':<6} {'Reward':<10} {'Episodes':<10}")
    print("-" * 55)
    
    for r in sorted(all_results, key=lambda x: (x["algo"], x["env"], x["seed"])):
        n_episodes = len(r.get("episode_rewards", []))
        print(f"{r['algo']:<10} {r['env']:<15} {r['seed']:<6} {r['final_eval_mean']:<10.3f} {n_episodes:<10}")
    
    print("\n" + "="*80)
    
    # Also save as markdown
    save_table_markdown(all_results, algos, envs)


def save_table_markdown(all_results, algos, envs):
    """Save results as a markdown table."""
    os.makedirs("results", exist_ok=True)
    
    with open("results/summary_table.md", "w") as f:
        f.write("# Experiment Results Summary\n\n")
        
        # Main comparison table
        f.write("## Final Evaluation Reward\n\n")
        f.write("| Algorithm | " + " | ".join(envs) + " |\n")
        f.write("|" + "---|" * (len(envs) + 1) + "\n")
        
        for algo in algos:
            row = f"| {algo} |"
            for env in envs:
                runs = [r for r in all_results if r["algo"] == algo and r["env"] == env]
                if runs:
                    vals = [r["final_eval_mean"] for r in runs]
                    mean, std = np.mean(vals), np.std(vals)
                    row += f" {mean:.3f} ± {std:.3f} |"
                else:
                    row += " - |"
            f.write(row + "\n")
        
        f.write("\n## Individual Runs\n\n")
        f.write("| Algorithm | Environment | Seed | Reward | Episodes |\n")
        f.write("|---|---|---|---|---|\n")
        
        for r in sorted(all_results, key=lambda x: (x["algo"], x["env"], x["seed"])):
            n_episodes = len(r.get("episode_rewards", []))
            f.write(f"| {r['algo']} | {r['env']} | {r['seed']} | {r['final_eval_mean']:.3f} | {n_episodes} |\n")
    
    print("Saved: results/summary_table.md")


# ============================================================================
# PLOTTING
# ============================================================================

def plot_learning_curves():
    """Plot episode rewards over training for each algorithm."""
    all_results = load_all_results()
    
    if not all_results:
        print("No results found in results/ directory")
        return
    
    os.makedirs("plots", exist_ok=True)
    
    # Group by environment
    envs = set(r["env"] for r in all_results)
    
    for env_key in envs:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for r in all_results:
            if r["env"] != env_key:
                continue
            
            timesteps = r.get("timesteps", [])
            rewards = r.get("episode_rewards", [])
            
            if not timesteps or not rewards:
                continue
            
            # Smooth with rolling average
            window = min(50, len(rewards) // 10 + 1)
            if len(rewards) > window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                x = timesteps[window-1:]
            else:
                smoothed = rewards
                x = timesteps
            
            label = f"{r['algo']} (seed={r['seed']})"
            ax.plot(x, smoothed, label=label, alpha=0.8)
        
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward (smoothed)")
        ax.set_title(f"Learning Curves: {env_key}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = f"plots/learning_curves_{env_key}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")
        plt.close()


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
            runs = [r for r in all_results if r["algo"] == algo_name and r["env"] == env_key]
            
            if not runs:
                continue
            
            # Average across seeds (simple version - just plot final eval)
            means = [r["final_eval_mean"] for r in runs]
            
            print(f"{env_key} | {algo_name}: {np.mean(means):.3f} ± {np.std(means):.3f}")
        
        # Bar plot of final performance
        algo_names = list(algos)
        means = []
        stds = []
        
        for algo_name in algo_names:
            runs = [r for r in all_results if r["algo"] == algo_name and r["env"] == env_key]
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
    parser.add_argument("--algo", type=str, choices=list(ALGORITHMS.keys()), 
                        help="Algorithm to use")
    parser.add_argument("--env", type=str, choices=list(ENVIRONMENTS.keys()),
                        help="Environment to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--plot", action="store_true",
                        help="Plot bar chart comparison of final results")
    parser.add_argument("--curves", action="store_true",
                        help="Plot learning curves over training")
    parser.add_argument("--table", action="store_true",
                        help="Print results table")
    parser.add_argument("--all", action="store_true",
                        help="Run all algorithm/environment/seed combinations")
    
    args = parser.parse_args()
    
    if args.table:
        print_table()
    elif args.curves:
        plot_learning_curves()
    elif args.plot:
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
