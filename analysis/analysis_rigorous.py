import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("results/multienv.csv")

print("="*70)
print("STEP A: CHECKING ASSUMPTIONS")
print("="*70)

strategies = ["epsilon_greedy", "softmax", "thompson"]

for env in df["environment"].unique():
    print(f"\n--- {env} ---")
    env_data = df[df["environment"] == env]

    # collect each strategy's total_reward values
    groups = {s: env_data[env_data["strategy"] == s]["total_reward"].values
              for s in strategies}

    # 1. Normality check (Shapiro-Wilk) for each strategy
    print("Normality (Shapiro-Wilk):")
    for s in strategies:
        values = groups[s]
        # Shapiro needs some variance; skip if all values identical
        if np.std(values) == 0:
            print(f"  {s:15s}: zero variance (all identical) → not normal")
        else:
            stat, p = stats.shapiro(values)
            verdict = "NOT normal" if p < 0.05 else "normal OK"
            print(f"  {s:15s}: p = {p:.4f} → {verdict}")

    # 2. Equal variance check (Levene) across the three strategies
    # (only include groups with some variance)
    valid_groups = [groups[s] for s in strategies if np.std(groups[s]) > 0]
    if len(valid_groups) >= 2:
        stat, p = stats.levene(*valid_groups)
        verdict = "NOT equal" if p < 0.05 else "equal OK"
        print(f"Equal variances (Levene): p = {p:.4f} → variances {verdict}")
    else:
        print("Equal variances (Levene): skipped (not enough varying groups)")

print("\n" + "="*70)
print("STEP B: NON-PARAMETRIC TESTS")
print("="*70)

for env in df["environment"].unique():
    print(f"\n--- {env} ---")
    env_data = df[df["environment"] == env]
    groups = {s: env_data[env_data["strategy"] == s]["total_reward"].values
              for s in strategies}

    # 1. Kruskal-Wallis: is there ANY difference among the three strategies?
    stat, p = stats.kruskal(groups["epsilon_greedy"],
                            groups["softmax"],
                            groups["thompson"])
    verdict = "YES, differ" if p < 0.05 else "no difference"
    print(f"Kruskal-Wallis (overall): p = {p:.2e} → {verdict}")

    # 2. Mann-Whitney U: pairwise comparisons
    print("Mann-Whitney U (pairwise):")
    pairs = [("thompson", "epsilon_greedy"),
             ("thompson", "softmax"),
             ("softmax", "epsilon_greedy")]
    for a, b in pairs:
        stat, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"  {a:15s} vs {b:15s}: p = {p:.2e} → {verdict}")

from statsmodels.stats.multitest import multipletests

print("\n" + "="*70)
print("STEP C: MULTIPLE-COMPARISON CORRECTION (Holm-Bonferroni)")
print("="*70)

# collect ALL pairwise p-values across all environments first
all_pvalues = []
all_labels = []

for env in df["environment"].unique():
    env_data = df[df["environment"] == env]
    groups = {s: env_data[env_data["strategy"] == s]["total_reward"].values
              for s in strategies}
    pairs = [("thompson", "epsilon_greedy"),
             ("thompson", "softmax"),
             ("softmax", "epsilon_greedy")]
    for a, b in pairs:
        stat, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        all_pvalues.append(p)
        all_labels.append(f"{env}: {a} vs {b}")

# apply Holm-Bonferroni correction to ALL tests together
reject, corrected_p, _, _ = multipletests(all_pvalues, alpha=0.05, method="holm")

print(f"\nTotal tests run: {len(all_pvalues)}")
print(f"{'Comparison':<45s} {'raw p':>10s} {'corrected p':>12s}  significant?")
print("-" * 85)
for label, raw, corr, rej in zip(all_labels, all_pvalues, corrected_p, reject):
    sig = "YES ✓" if rej else "no"
    print(f"{label:<45s} {raw:>10.2e} {corr:>12.2e}  {sig}")

import statsmodels.api as sm
from statsmodels.formula.api import ols

print("\n" + "="*70)
print("STEP D: TWO-WAY ANALYSIS (Strategy x Environment interaction)")
print("="*70)

# Standardize total_reward WITHIN each environment (z-score)
# so different reward scales don't dominate the analysis
df_std = df.copy()
df_std["reward_z"] = df_std.groupby("environment")["total_reward"].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

# Two-way ANOVA on the standardized data
model = ols("reward_z ~ C(strategy) + C(environment) + C(strategy):C(environment)",
            data=df_std).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nTwo-way ANOVA (on within-environment standardized rewards):")
print(anova_table.round(4))

print("\nInterpretation:")
for factor in anova_table.index:
    if factor == "Residual":
        continue
    p = anova_table.loc[factor, "PR(>F)"]
    sig = "significant" if p < 0.05 else "not significant"
    nice_name = (factor.replace("C(strategy):C(environment)", "INTERACTION")
                       .replace("C(strategy)", "Strategy")
                       .replace("C(environment)", "Environment"))
    print(f"  {nice_name:15s}: p = {p:.2e} → {sig}")

    print("\n" + "="*70)
print("STEP E: BOOTSTRAP CONFIDENCE INTERVALS")
print("="*70)

def bootstrap_ci(data, n_boot=10000, ci=95):
    """Compute a bootstrap confidence interval for the mean."""
    boot_means = []
    n = len(data)
    for _ in range(n_boot):
        # resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

np.random.seed(42)  # reproducibility

for env in df["environment"].unique():
    print(f"\n--- {env} ---")
    env_data = df[df["environment"] == env]
    for s in strategies:
        values = env_data[env_data["strategy"] == s]["total_reward"].values
        mean, lower, upper = bootstrap_ci(values)
        print(f"  {s:15s}: mean = {mean:10.1f}  95% CI = [{lower:10.1f}, {upper:10.1f}]")