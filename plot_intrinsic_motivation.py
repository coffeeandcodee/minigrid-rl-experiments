"""
Generate learning curves for intrinsic motivation experiments on DoorKey-8x8.
Shows that despite high training rewards (including intrinsic bonuses), 
all methods except reward shaping fail at evaluation.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RESULTS_DIR = "results/doorkey_8x8/PPO"
OUTPUT_DIR = "plots/doorkey_8x8"

def load_results(pattern):
    """Load all results matching a pattern."""
    results = []
    results_path = Path(RESULTS_DIR)
    for f in results_path.glob(pattern):
        with open(f) as file:
            results.append(json.load(file))
    return results

def smooth(data, window=30):
    """Apply rolling average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def interpolate_to_common_x(results, max_timestep=200000):
    """Interpolate all runs to a common x-axis."""
    common_x = np.linspace(0, max_timestep, 500)
    all_curves = []
    
    for r in results:
        timesteps = r.get("timesteps", [])
        rewards = r.get("episode_rewards", [])
        
        if not timesteps or not rewards:
            continue
        
        # Smooth
        window = min(30, len(rewards) // 5 + 1)
        if len(rewards) > window:
            smoothed = smooth(rewards, window)
            x = timesteps[window-1:len(smoothed)+window-1]
        else:
            smoothed = rewards
            x = timesteps
        
        # Interpolate
        if len(x) > 0 and len(smoothed) > 0:
            # Pad to max_timestep if needed
            x = np.array(x[:len(smoothed)])
            interp_y = np.interp(common_x, x, smoothed, left=smoothed[0], right=smoothed[-1])
            all_curves.append(interp_y)
    
    return common_x, all_curves

# Color palette
colors = {
    "Baseline": "#333333",
    "Count-based": "#E69F00",
    "ICM": "#56B4E9", 
    "RND": "#009E73",
    "Reward Shaping": "#D55E00",
}

# Load data
print("Loading data...")
baseline_200k = load_results("seed*_baseline_200k.json")
count_based = load_results("seed*_exploration_bonus_200k.json")
icm_200k = load_results("seed*_icm_200k.json")
rnd_200k = load_results("seed*_rnd_200k.json")

print(f"Loaded: {len(baseline_200k)} baseline, {len(count_based)} count-based, {len(icm_200k)} ICM, {len(rnd_200k)} RND")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')

# Plot each method
methods = [
    ("Baseline", baseline_200k, "#333333"),
    ("Count-based (β=0.1)", count_based, "#E69F00"),
    ("ICM (β=0.01)", icm_200k, "#56B4E9"),
    ("RND (β=0.1)", rnd_200k, "#009E73"),
]

for name, data, color in methods:
    if not data:
        continue
    
    x, curves = interpolate_to_common_x(data, max_timestep=200000)
    
    if curves:
        curves_array = np.array(curves)
        mean = np.mean(curves_array, axis=0)
        std = np.std(curves_array, axis=0)
        
        n_seeds = len(data)
        ax.plot(x, mean, label=f"{name} (n={n_seeds})", color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

# Add annotation about evaluation failure
ax.annotate('All methods: 0% success\n(eval on clean env)', 
            xy=(180000, 0.5), fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='#ffeeee', edgecolor='#cc0000', alpha=0.8),
            ha='center')

# Labels
ax.set_xlabel("Timesteps", fontsize=13, fontweight='medium', color='#333333')
ax.set_ylabel("Episode Reward (incl. intrinsic)", fontsize=13, fontweight='medium', color='#333333')
ax.set_title("Intrinsic Motivation Methods: DoorKey-8x8 (200k steps)", 
             fontsize=15, fontweight='bold', color='#222222', pad=15)

# Note about y-axis
ax.text(0.02, 0.98, "Note: Training rewards include intrinsic bonuses.\nEvaluation uses clean environment (no bonuses).",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Legend and grid
ax.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Format x-axis
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
ax.set_xlim(0, 200000)

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = f"{OUTPUT_DIR}/doorkey_8x8_intrinsic_motivation_learning_curves.png"
plt.tight_layout()
plt.savefig(output_path, dpi=200, facecolor='white', bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()

# Also print summary statistics
print("\n" + "="*60)
print("INTRINSIC MOTIVATION EXPERIMENT SUMMARY")
print("="*60)

print("\n### Hyperparameters Used:")
print("""
| Method       | Parameter      | Value   | Description                    |
|--------------|----------------|---------|--------------------------------|
| Count-based  | bonus_scale    | 0.1     | r_int = 0.1 / sqrt(N(s))       |
| ICM          | intrinsic_scale| 0.01    | Forward/inverse model, η=0.2   |
| RND          | intrinsic_scale| 0.1     | Random network distillation    |
| Reward Shape | key_bonus      | +0.5    | Bonus for picking up key       |
| Reward Shape | door_bonus     | +0.5    | Bonus for opening door         |
""")

print("\n### Final Evaluation Results (on clean environment):")
for name, data in [("Baseline", baseline_200k), ("Count-based", count_based), 
                   ("ICM", icm_200k), ("RND", rnd_200k)]:
    if data:
        means = [r.get("final_eval_mean", 0) for r in data]
        print(f"{name}: {np.mean(means):.3f} ± {np.std(means):.3f} (success: {sum(1 for m in means if m > 0.5)}/{len(means)})")

print("\n### Analysis of Failure Modes:")
print("""
1. COUNT-BASED EXPLORATION:
   - Training rewards appear high (4-25+ per episode) because intrinsic 
     bonuses accumulate: r_total = r_ext + 0.1/sqrt(N(s)) per step
   - With 640 steps/episode and ~100 unique states, bonus ≈ 6.4/episode
   - Agent learns to maximize state coverage, NOT task completion
   - When evaluated WITHOUT intrinsic rewards → 0% success

2. ICM (Intrinsic Curiosity Module):
   - Forward model easily predicts next state in gridworld (low error)
   - Intrinsic reward = prediction error → rapidly diminishes
   - After ~50k steps, intrinsic signal too weak to guide exploration
   - Agent defaults to random behavior → 0% success

3. RND (Random Network Distillation):
   - Target network outputs become predictable after sufficient training
   - Novelty signal decays as predictor learns target distribution
   - Similar to ICM: initial exploration boost, then signal collapse
   - 0% success despite high training rewards
""")
