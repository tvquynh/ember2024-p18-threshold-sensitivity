"""Regenerate every inferential number printed in the manuscript, from the
per-seed values stored in results_aggregated/*/summary.json.

This file is the authoritative statistical companion to the revised
manuscript. It supersedes the archived results_aggregated/*/stats.json,
which was produced by an earlier aggregation pass over a larger threshold
grid (9 coarse / 14 fine, including thresholds below the baseline) that the
study later dropped; those files are retained only for provenance and their
`correction` / `family_size` / `k` fields do NOT describe the analysis
reported in the paper.

What is recomputed here:
  * Friedman chi-square per classifier on the 6-threshold coarse grid
  * Friedman chi-square for the 11-threshold LightGBM fine grid
  * Wilcoxon signed-rank (exact) vs the baseline for every non-baseline
    threshold, with Holm-Bonferroni correction at the family sizes the
    paper states (10 for the fine grid; 5 / 3 for the per-classifier
    weighting families)
  * Cohen's d on seed-paired differences
  * Spearman rho between threshold and mean challenge TPR (Random Forest)

Run from the repository root:
    python revision1_analyses/recompute_paper_stats.py

Writes revision1_analyses/recomputed_stats_revision1.json and prints a
human-readable trace.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_aggregated"
OUT = Path(__file__).resolve().parent / "recomputed_stats_revision1.json"

METRIC = "TPR@1.0%FPR"
COARSE = ["0.065", "0.100", "0.150", "0.200", "0.300", "0.500"]
FINE = ["0.065", "0.080", "0.100", "0.120", "0.150", "0.180",
        "0.200", "0.250", "0.300", "0.400", "0.500"]


def holm(pvals):
    """Holm-Bonferroni step-down, returning adjusted p in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(p[idx] * (n - rank), 1.0))
        adj[idx] = running
    return adj


def cohen_d_paired(diff):
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def vals(entry):
    return np.asarray(entry["challenge"][METRIC]["values"], dtype=float) * 100.0


out = {"note": "Authoritative recomputation for the revised manuscript; "
               "supersedes results_aggregated/*/stats.json (earlier, larger grid).",
       "metric": METRIC, "unit": "percentage points", "n_seeds": 10}

# ---------- Friedman, coarse grid ----------
cc = json.loads((RES / "cross_classifier" / "summary.json").read_text())
out["friedman_coarse"] = {}
print("Friedman, coarse grid (k = 6, n = 10):")
for clf in ["LightGBM", "XGBoost", "RandomForest", "MLP"]:
    mat = [vals(cc[clf][T]) for T in COARSE]
    chi2, p = friedmanchisquare(*mat)
    out["friedman_coarse"][clf] = {"chi2": float(chi2), "p": float(p), "k": 6, "n": 10}
    print(f"  {clf:13s} chi2 = {chi2:7.3f}   p = {p:.3e}")

# ---------- Friedman, fine grid ----------
fg = json.loads((RES / "lgbm_fine" / "summary.json").read_text())
mat = [vals(fg["LightGBM"][T]) for T in FINE]
chi2, p = friedmanchisquare(*mat)
out["friedman_fine"] = {"classifier": "LightGBM", "chi2": float(chi2),
                        "p": float(p), "k": len(FINE), "n": 10}
print(f"\nFriedman, LightGBM fine grid (k = {len(FINE)}, n = 10): chi2 = {chi2:.3f}  p = {p:.3e}")

# ---------- Fine-grid pairwise vs baseline (Table 3) ----------
base = vals(fg["LightGBM"]["0.065"])
nb = [T for T in FINE if T != "0.065"]
raw, rows = [], []
for T in nb:
    v = vals(fg["LightGBM"][T])
    d = v - base
    _, pr = wilcoxon(v, base, alternative="two-sided", method="exact")
    raw.append(pr)
    rows.append({"T": float(T), "mean_pct": float(v.mean()), "std_pct": float(v.std(ddof=1)),
                 "delta_pp": float(d.mean()), "cohens_d": cohen_d_paired(d), "p_raw": float(pr)})
adj = holm(raw)
for r, a in zip(rows, adj):
    r["p_holm"] = float(a)
    r["holm_significant"] = bool(a < 0.05)
out["fine_grid_vs_baseline"] = {
    "baseline_mean_pct": float(base.mean()), "baseline_std_pct": float(base.std(ddof=1)),
    "family_size": len(nb), "exact_wilcoxon_floor": 2 / 2 ** 10, "rows": rows}
print(f"\nTable 3 (family size {len(nb)}, exact-Wilcoxon floor {2/2**10:.5f}):")
print(f"  baseline T=0.065: mean {base.mean():.2f}%  std {base.std(ddof=1):.3f}pp")
for r in rows:
    print(f"  T={r['T']:.3f}  mean {r['mean_pct']:6.2f}  std {r['std_pct']:.2f}  "
          f"delta {r['delta_pp']:+6.2f}  d {r['cohens_d']:+6.2f}  "
          f"p_raw {r['p_raw']:.4f}  p_holm {r['p_holm']:.4f}{' *' if r['holm_significant'] else ''}")

# ---------- Weighting families (Table 4) ----------
wt = json.loads((RES / "weighting" / "summary.json").read_text())
out["weighting"] = {}
print("\nTable 4 (per-classifier Holm families):")
for clf in ["LightGBM", "XGBoost"]:
    if clf not in wt:
        continue
    uni = vals(wt[clf]["uniform"])
    names = [k for k in wt[clf] if k != "uniform"]
    raw, rws = [], []
    for k in names:
        v = vals(wt[clf][k])
        d = v - uni
        _, pr = wilcoxon(v, uni, alternative="two-sided", method="exact")
        raw.append(pr)
        rws.append({"strategy": k, "mean_pct": float(v.mean()), "delta_pp": float(d.mean()),
                    "cohens_d": cohen_d_paired(d), "p_raw": float(pr)})
    adj = holm(raw)
    for r, a in zip(rws, adj):
        r["p_holm"] = float(a)
        r["holm_significant"] = bool(a < 0.05)
    out["weighting"][clf] = {"uniform_mean_pct": float(uni.mean()),
                             "family_size": len(names), "rows": rws}
    print(f"  {clf} (F = {len(names)}, uniform {uni.mean():.2f}%)")
    for r in sorted(rws, key=lambda x: x["delta_pp"], reverse=True):
        print(f"    {r['strategy']:13s} mean {r['mean_pct']:6.2f}  delta {r['delta_pp']:+7.2f}  "
              f"d {r['cohens_d']:+6.2f}  p_raw {r['p_raw']:.4f}  p_holm {r['p_holm']:.4f}"
              f"{' *' if r['holm_significant'] else ''}")

# ---------- Random Forest monotonicity ----------
T_num = [float(T) for T in COARSE]
rf_means = [float(vals(cc["RandomForest"][T]).mean()) for T in COARSE]
rho, prho = spearmanr(T_num, rf_means)
out["random_forest_monotonicity"] = {"thresholds": T_num, "mean_pct": rf_means,
                                     "spearman_rho": float(rho), "p": float(prho)}
print(f"\nRandom Forest: means {[round(m, 2) for m in rf_means]}  ->  Spearman rho = {rho:.4f}")

OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}")
