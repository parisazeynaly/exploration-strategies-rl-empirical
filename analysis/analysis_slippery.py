import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# Load both datasets
slippery = pd.read_csv("results/slippery.csv")
determ = pd.read_csv("results/multienv.csv")

strategies = ["epsilon_greedy", "softmax", "thompson"]
labels = ["ε-greedy", "softmax", "thompson"]

# --- Chart 1: Deterministic vs Slippery final success (4x4 and 8x8) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, map_size in zip(axes, ["4x4", "8x8"]):
    # deterministic final success for this map
    det_env = f"FrozenLake-{map_size}"
    det_means = [determ[(determ["environment"] == det_env) &
                        (determ["strategy"] == s)]["final_reward"].mean()
                 for s in strategies]
    # slippery final success for this map
    slip_means = [slippery[(slippery["map_size"] == map_size) &
                           (slippery["strategy"] == s)]["final_success"].mean()
                  for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35
    ax.bar(x - width/2, det_means, width, label="Deterministic", color="steelblue")
    ax.bar(x + width/2, slip_means, width, label="Slippery", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final success rate")
    ax.set_title(f"FrozenLake {map_size}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Deterministic vs. Slippery: Final Success Rate", fontsize=14)
plt.tight_layout()
plt.savefig("figures/determ_vs_slippery.png", dpi=150, bbox_inches="tight")
print("Saved figures/determ_vs_slippery.png")
plt.close()


# --- Chart 2: Thompson parameter sweep on slippery 4x4 ---
sweep = pd.read_csv("results/slippery_thompson_sweep.csv")

grouped = sweep.groupby("scale")["final_success"]
means = grouped.mean()
stderr = grouped.std() / np.sqrt(grouped.count())

plt.figure(figsize=(9, 6))
plt.errorbar(means.index, means.values, yerr=1.96 * stderr.values,
             marker="o", capsize=5, linewidth=2, color="purple")
plt.axhline(y=0.709, color="gray", linestyle="--",
            label="ε-greedy baseline (0.709)")
plt.xlabel("Thompson uncertainty scale")
plt.ylabel("Final success rate (slippery 4x4)")
plt.title("Thompson Sampling: More Exploration Helps Under Stochasticity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/thompson_slippery_sweep.png", dpi=150, bbox_inches="tight")
print("Saved figures/thompson_slippery_sweep.png")
plt.close()