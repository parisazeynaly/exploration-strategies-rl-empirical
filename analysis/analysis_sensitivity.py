import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/sensitivity.csv")
os.makedirs("figures", exist_ok=True)

print("Sensitivity data overview:")
print(df.groupby(["strategy", "param"])["total_reward"].agg(["mean", "std"]).round(1))

# ---- Plot: performance vs parameter for each strategy ----
strategies = df["strategy"].unique()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, strategy in zip(axes, strategies):
    strat_data = df[df["strategy"] == strategy]

    # mean and standard error across seeds, for each parameter value
    grouped = strat_data.groupby("param")["total_reward"]
    means = grouped.mean()
    stderr = grouped.std() / np.sqrt(grouped.count())

    ax.errorbar(means.index, means.values, yerr=1.96 * stderr.values,
                marker="o", capsize=5, linewidth=2)
    ax.set_title(f"{strategy}")
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Total reward (area under curve)")
    ax.grid(True, alpha=0.3)

plt.suptitle("Parameter Sensitivity on FrozenLake 8x8", fontsize=14)
plt.tight_layout()
plt.savefig("figures/sensitivity.png", dpi=150, bbox_inches="tight")
print("\nSaved figures/sensitivity.png")
plt.close()