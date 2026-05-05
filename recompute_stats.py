"""Recompute Friedman + pairwise Wilcoxon + Cohen's d on the 6-threshold
coarse grid and 11-threshold fine grid from the aggregated per-run JSON
outputs in `results_aggregated/`.

Run from repository root:
    python recompute_stats.py

The numbers printed here are the same numbers that appear in Section 5.2
("Cross-classifier threshold sensitivity") and Section 5.3 ("Sweet zone
for LightGBM and variance stabilisation") of the manuscript and Table 2
of the paper.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import json
import numpy as np
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent / "results_aggregated"
COARSE_T = ["0.065", "0.100", "0.150", "0.200", "0.300", "0.500"]
FINE_T = ["0.065", "0.080", "0.100", "0.120", "0.150", "0.180",
          "0.200", "0.250", "0.300", "0.400", "0.500"]

def cohen_d_paired(diff):
    """Cohen's d for paired samples = mean(diff) / std(diff, ddof=1)."""
    diff = np.asarray(diff)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0

def holm_bonferroni(p_values):
    """Return Holm-corrected p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_holm = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p[idx] * (n - rank)
        running_max = max(running_max, min(adj, 1.0))
        p_holm[idx] = running_max
    return p_holm

# ---- 1. Cross-classifier Friedman (6 thresholds × 10 seeds × 4 classifiers) ----
print("=" * 72)
print("CROSS-CLASSIFIER (challenge TPR@1%FPR, 6 thresholds, 10 seeds)")
print("=" * 72)

cc = json.loads((ROOT / "cross_classifier" / "summary.json").read_text())
metric_key = "TPR@1.0%FPR"

friedman_results = {}
for clf in ["LightGBM", "XGBoost", "RandomForest", "MLP"]:
    # Build seed × threshold matrix
    matrix = []
    for T in COARSE_T:
        vals = cc[clf][T]["challenge"][metric_key]["values"]
        matrix.append(vals)
    matrix = np.array(matrix)  # shape (k=6, n=10)
    # Friedman wants samples for each measurement -> pass each threshold as arg
    stat, p = friedmanchisquare(*matrix)
    friedman_results[clf] = {"stat": float(stat), "p": float(p), "k": 6, "n": 10}
    print(f"  {clf:12s}  chi2 = {stat:7.3f}  p = {p:.3e}  (k=6, n=10)")

print()

# ---- 2. LightGBM fine-grid Friedman (11 thresholds × 10 seeds) ----
print("=" * 72)
print("LIGHTGBM FINE GRID (challenge TPR@1%FPR, 11 thresholds, 10 seeds)")
print("=" * 72)

fg = json.loads((ROOT / "lgbm_fine" / "summary.json").read_text())
fg_clf = list(fg.keys())[0]  # likely "LightGBM"
fg_data = fg[fg_clf]
matrix_fine = []
for T in FINE_T:
    if T in fg_data:
        vals = fg_data[T]["challenge"][metric_key]["values"]
        matrix_fine.append(vals)
    else:
        print(f"  WARNING: T={T} missing from fine-grid summary")
matrix_fine = np.array(matrix_fine)
print(f"  Matrix shape: {matrix_fine.shape}  (expected: 11 × 10)")
stat_fine, p_fine = friedmanchisquare(*matrix_fine)
print(f"  Fine-grid Friedman chi2 = {stat_fine:.3f}  p = {p_fine:.3e}  (k={matrix_fine.shape[0]}, n=10)")

print()

# ---- 3. Pairwise Wilcoxon + Cohen's d for fine grid (10 non-baseline) ----
print("=" * 72)
print("LIGHTGBM FINE-GRID PAIRWISE (vs T_base = 0.065, Holm-Bonferroni on n=10)")
print("=" * 72)

baseline_vals = np.array(fg_data["0.065"]["challenge"][metric_key]["values"])
baseline_pct = 100 * baseline_vals  # convert fraction → percent
baseline_mean = baseline_pct.mean()
baseline_std = baseline_pct.std(ddof=1)
print(f"  Baseline T=0.065:  mean = {baseline_mean:.2f}%  std = {baseline_std:.3f}pp")
print()

print(f"  {'T':6s} {'Mean (%)':>9s} {'Std (%)':>8s} {'Δ (pp)':>8s} {'Cohen d':>8s} {'p (raw)':>10s} {'p (Holm)':>10s} {'sig?':>5s}")
non_baseline = [t for t in FINE_T if t != "0.065"]
raw_p = []
deltas, ds, means, stds = [], [], [], []
for T in non_baseline:
    vals = np.array(fg_data[T]["challenge"][metric_key]["values"])
    pct = 100 * vals
    diff_pct = pct - baseline_pct
    mean_T = pct.mean()
    std_T = pct.std(ddof=1)
    delta = diff_pct.mean()
    d = cohen_d_paired(diff_pct)
    try:
        w_stat, p = wilcoxon(pct, baseline_pct, zero_method="wilcox")
    except ValueError:
        p = 1.0
    raw_p.append(p)
    means.append(mean_T)
    stds.append(std_T)
    deltas.append(delta)
    ds.append(d)

p_holm = holm_bonferroni(raw_p)
sig = p_holm < 0.05

for i, T in enumerate(non_baseline):
    flag = "*" if sig[i] else " "
    print(f"  {T:6s} {means[i]:>9.2f} {stds[i]:>8.3f} {deltas[i]:>+8.2f} {ds[i]:>+8.2f} {raw_p[i]:>10.4f} {p_holm[i]:>10.4f} {flag:>5s}")

print()
print(f"Family size for Holm-Bonferroni: {len(non_baseline)} (= 10 non-baseline thresholds)")
