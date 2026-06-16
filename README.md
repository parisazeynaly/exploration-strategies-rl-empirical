# An Empirical Comparison of Classical and Bayesian Exploration Strategies in Reinforcement Learning

**Parisa Zeinaliashtiyani** — University of Naples Federico II  
Department of Electrical Engineering and Information Technology  
`p.zeinaliahstiyani@unina.studenti.it`

---

## Overview

This repository contains the full code and results for an empirical study of exploration strategies in tabular reinforcement learning. We compare three action-selection strategies within the Q-learning framework across environments of varying size and stochasticity:

| Strategy | Type | Key idea |
|---|---|---|
| Epsilon-greedy | Classical | Random with prob ε, greedy otherwise |
| Softmax (Boltzmann) | Classical | Probability proportional to exp(Q/τ) |
| Thompson Sampling | Bayesian | Sample from posterior N(μ, σ²) per action |

**Research question:** Does Bayesian uncertainty-based exploration converge faster and perform more stably than classical strategies — and does environment stochasticity modulate this advantage?

---

## Experimental Design

A 2×2 factorial design across grid size and transition stochasticity:

```
              Deterministic     Stochastic
              (is_slippery=F)   (is_slippery=T)
              ─────────────────────────────────
4×4 grid   │   Condition 1   │   Condition 2  │
8×8 grid   │   Condition 3   │   Condition 4  │
```

- **30 independent runs** per condition (seeds 0–29)
- **5,000 episodes** for 4×4 · **15,000 episodes** for 8×8
- All strategies share identical Q-learning hyperparameters (α=0.1, γ=0.99)

---

## Repository Structure

```
.
├── rl_exploration_comparison.py   # main experiment script
├── results/
│   ├── curves_4x4_det.png         # learning curves — 4×4 deterministic
│   ├── curves_4x4_sto.png         # learning curves — 4×4 stochastic
│   ├── curves_8x8_det.png         # learning curves — 8×8 deterministic
│   ├── curves_8x8_sto.png         # learning curves — 8×8 stochastic
│   ├── convergence_scatter.png    # convergence speed vs stability
│   ├── all_metrics.json           # full metric table (all conditions)
│   └── statistical_tests.json    # Kruskal-Wallis + Mann-Whitney results
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/<your-username>/rl-exploration-comparison
cd rl-exploration-comparison
pip install gymnasium numpy scipy matplotlib
```

---

## Running the Experiment

```bash
python rl_exploration_comparison.py
```

This runs all 4 conditions × 3 strategies × 30 seeds and saves plots and metrics to `results/`. Expected runtime: 10–30 minutes depending on hardware.

To test a single condition first, edit the `CONFIG` dict at the top of the script:

```python
CONFIG = {
    "4x4": {"episodes": 5_000, "map": "4x4"},
    # "8x8": {"episodes": 15_000, "map": "8x8"},   # comment out to skip
}
```

---

## Methods

### Q-learning update (shared by all strategies)

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

### Action selection

**Epsilon-greedy**
```
if rand() < ε  →  random action
else           →  argmax_a Q(s,a)
ε decays linearly: 1.0 → 0.01 over training
```

**Softmax**
```
P(a) = exp(Q(s,a) / τ) / Σ exp(Q(s,a') / τ)
τ decays linearly: 1.0 → 0.01 over training
```

**Thompson Sampling**
```
sample_a ~ N(μ(s,a), σ²(s,a))
action   =  argmax over samples
σ²(s,a) =  1 / (n(s,a) + 1)     # shrinks as action is visited more
```

### Evaluation metrics

- **Mean cumulative reward** — success rate per episode, averaged over 30 runs
- **Convergence speed** — first episode where rolling mean stays ≥ 0.7 for 100 consecutive episodes
- **Policy stability** — variance of mean reward over the final 20% of training
- **95% confidence intervals** — bootstrap resampling (n=2,000)

### Statistical tests

- **Kruskal-Wallis** across all 3 strategies per condition (non-parametric, no normality assumption)
- **Mann-Whitney U** pairwise with Bonferroni correction (α = 0.05/3 ≈ 0.017)
- **Effect size r** = 1 − 2U/(n₁n₂)

---

## Results

> Full results and plots are in `results/` after running the script.  
> The paper with complete analysis is available at: `[link to paper / arXiv]`

---

## Hyperparameters

| Parameter | 4×4 | 8×8 |
|---|---|---|
| Episodes | 5,000 | 15,000 |
| Learning rate α | 0.1 | 0.1 |
| Discount factor γ | 0.99 | 0.99 |
| Initial ε / τ | 1.0 | 1.0 |
| Final ε / τ | 0.01 | 0.01 |
| Decay | linear | linear |
| Independent runs | 30 | 30 |
| Random seeds | 0–29 | 0–29 |

---

## Environment

[FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) from [Gymnasium](https://gymnasium.farama.org/) (Towers et al., 2023).

A grid-world where the agent navigates from start (S) to goal (G) avoiding holes (H). `is_slippery=True` adds stochastic transitions: with probability 2/3 the agent slips perpendicular to the intended direction.

```
4×4 map          8×8 map (excerpt)
S F F F          S F F F F F F F
F H F H          F F F F F H F F
F F F H          F F H F F F F F
H F F G          ...         F G
```

---

## Citation

If you use this code, please cite:

```bibtex

```

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*, 25(3-4), 285–294.
- Russo, D. et al. (2018). A tutorial on Thompson Sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
- Towers, M. et al. (2023). Gymnasium. Zenodo. https://doi.org/10.5281/zenodo.8127025

---

*M.Sc. Data Science & Engineering — University of Naples Federico II*
