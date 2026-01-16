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


def save_results(results):
    """Save results to JSON file in env/algo subdirectory."""
    env_key = results.get("env", "unknown")
    algo_name = results.get("algo", "unknown")
    seed = results.get("seed", 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_dir = os.path.join(RESULTS_DIR, env_key, algo_name)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"seed{seed}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {filepath}")


def load_all_results():
    """Load all results from the results directory (including subdirectories)."""
    all_results = []
    if not os.path.exists(RESULTS_DIR):
        return all_results
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(RESULTS_DIR):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                with open(filepath) as f:
                    all_results.append(json.load(f))
    return all_results


def run_all_experiments():
    """Run all algorithm/environment/seed combinations."""
    for algo_name in ALGORITHMS:
        for env_key in ENVIRONMENTS:
            for seed in SEEDS:
                results = run_single_experiment(algo_name, env_key, seed)
                save_results(results)


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
    """Plot episode rewards over training for each algorithm with mean ± std."""
    all_results = load_all_results()
    
    if not all_results:
        print("No results found in results/ directory")
        return
    
    os.makedirs("plots", exist_ok=True)
    
    # Group by environment
    envs = set(r["env"] for r in all_results)
    
    # Sophisticated color palette (colorblind-friendly)
    algo_colors = {
        "PPO": "#4C72B0",   # Steel blue
        "A2C": "#55A868",   # Sage green
        "DQN": "#C44E52",   # Muted red
    }
    
    # Nice display names for environments
    env_display_names = {
        "empty_5x5": "Empty-5×5",
        "empty_8x8": "Empty-8×8",
        "doorkey_5x5": "DoorKey-5×5",
        "doorkey_6x6": "DoorKey-6×6",
    }
    
    for env_key in envs:
        # Set up figure with clean style
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        
        # Group results by algorithm
        algos = set(r["algo"] for r in all_results if r["env"] == env_key)
        
        # Store data for annotations
        algo_data = {}
        
        for algo in sorted(algos):
            algo_runs = [r for r in all_results if r["env"] == env_key and r["algo"] == algo]
            
            if not algo_runs:
                continue
            
            # Interpolate all runs to common x-axis
            max_timestep = min(r["timesteps"][-1] for r in algo_runs if r.get("timesteps"))
            common_x = np.linspace(0, max_timestep, 500)
            
            all_smoothed = []
            for r in algo_runs:
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
                
                # Interpolate to common x-axis
                interp_y = np.interp(common_x, x, smoothed)
                all_smoothed.append(interp_y)
            
            if not all_smoothed:
                continue
            
            # Calculate mean and std
            all_smoothed = np.array(all_smoothed)
            mean = np.mean(all_smoothed, axis=0)
            std = np.std(all_smoothed, axis=0)
            
            # Store for annotations
            algo_data[algo] = {"x": common_x, "mean": mean, "std": std}
            
            # Plot mean line and shaded std region
            color = algo_colors.get(algo, "#666666")
            n_seeds = len(algo_runs)
            ax.plot(common_x, mean, label=f"{algo} (n={n_seeds})", color=color, 
                    linewidth=2.5, zorder=3)
            ax.fill_between(common_x, mean - std, mean + std, alpha=0.15, 
                            color=color, zorder=2)
        
        # Add annotation for DQN collapse on empty_5x5
        if env_key == "empty_5x5" and "DQN" in algo_data:
            dqn_mean = algo_data["DQN"]["mean"]
            dqn_x = algo_data["DQN"]["x"]
            # Find collapse point (where it drops below 0.5 after being above 0.7)
            peak_idx = np.argmax(dqn_mean)
            if dqn_mean[peak_idx] > 0.7:
                # Find where it drops
                collapse_idx = peak_idx
                for i in range(peak_idx, len(dqn_mean)):
                    if dqn_mean[i] < 0.4:
                        collapse_idx = i
                        break
                if collapse_idx > peak_idx:
                    collapse_x = dqn_x[collapse_idx]
                    collapse_y = dqn_mean[collapse_idx]
                    ax.annotate('DQN collapse', 
                                xy=(collapse_x, collapse_y),
                                xytext=(collapse_x + 5000, collapse_y + 0.25),
                                fontsize=10, color='#C44E52',
                                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
                                fontweight='medium')
        
        # Subtle grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#cccccc')
        ax.set_axisbelow(True)
        
        # Labels with better typography
        ax.set_xlabel("Timesteps", fontsize=13, fontweight='medium', color='#333333')
        ax.set_ylabel("Episode Reward", fontsize=13, fontweight='medium', color='#333333')
        
        # Title
        display_name = env_display_names.get(env_key, env_key)
        ax.set_title(f"Learning Curves: {display_name}", fontsize=15, 
                     fontweight='bold', color='#222222', pad=15)
        
        # Legend with frame
        legend = ax.legend(fontsize=11, loc='lower right', frameon=True, 
                          fancybox=True, shadow=False, framealpha=0.9,
                          edgecolor='#cccccc')
        legend.get_frame().set_linewidth(0.5)
        
        # Axis limits and ticks
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, None)
        ax.tick_params(axis='both', labelsize=11, colors='#333333')
        
        # Format x-axis with thousands separator
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Save with high quality
        plot_path = f"plots/{env_key}_learning_curves.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, facecolor='white', edgecolor='none',
                    bbox_inches='tight')
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
    
    # Sophisticated color palette
    algo_colors = {
        "PPO": "#4C72B0",
        "A2C": "#55A868", 
        "DQN": "#C44E52",
    }
    
    # Nice display names
    env_display_names = {
        "empty_5x5": "Empty-5×5",
        "empty_8x8": "Empty-8×8",
        "doorkey_5x5": "DoorKey-5×5",
        "doorkey_6x6": "DoorKey-6×6",
    }
    
    for env_key in envs:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        
        for algo_name in algos:
            # Get all runs for this algo/env
            runs = [r for r in all_results if r["algo"] == algo_name and r["env"] == env_key]
            
            if not runs:
                continue
            
            # Average across seeds
            means = [r["final_eval_mean"] for r in runs]
            print(f"{env_key} | {algo_name}: {np.mean(means):.3f} ± {np.std(means):.3f}")
        
        # Bar plot of final performance
        algo_names = sorted(list(algos))  # Sort for consistent ordering
        means = []
        stds = []
        colors = []
        
        for algo_name in algo_names:
            runs = [r for r in all_results if r["algo"] == algo_name and r["env"] == env_key]
            if runs:
                vals = [r["final_eval_mean"] for r in runs]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
            colors.append(algo_colors.get(algo_name, "#666666"))
        
        x = np.arange(len(algo_names))
        bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, 
                      edgecolor='white', linewidth=1.5, alpha=0.9,
                      error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': '#333333'})
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.03,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=12,
                    fontweight='medium', color='#333333')
        
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, fontsize=12, fontweight='medium')
        ax.set_ylabel("Final Evaluation Reward", fontsize=13, fontweight='medium', color='#333333')
        
        display_name = env_display_names.get(env_key, env_key)
        ax.set_title(f"Algorithm Comparison: {display_name}", fontsize=15, 
                     fontweight='bold', color='#222222', pad=15)
        
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='both', labelsize=11, colors='#333333')
        
        # Subtle grid (horizontal only)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#cccccc')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        os.makedirs("plots", exist_ok=True)
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
        save_results(results)
    else:
        print(__doc__)
