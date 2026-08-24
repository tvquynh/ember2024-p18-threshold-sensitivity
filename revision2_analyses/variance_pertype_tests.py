#!/usr/bin/env python3
"""Variance-equality and per-file-type significance tests (JISA revision 1).

Two claims in the manuscript rested on descriptive statistics alone. This
script attaches a test to each, using only the per-seed values already
stored in ``results_aggregated/`` -- no model is retrained.

  (A) Section 5.3, "5-8x smaller seed-to-seed standard deviation".
      A ratio of two sample standard deviations at n = 10 is itself noisy,
      so the reduction is tested with the Pitman-Morgan paired
      variance-equality test (Pitman 1939; Morgan 1939): for correlated
      x, y the Pearson correlation between (x + y) and (x - y) is zero
      under H0: var(x) = var(y). Holm-Bonferroni correction is applied
      across the ten non-baseline LightGBM fine-grid thresholds.

  (B) Section 5.7 and Table 9, per-file-type specialists.
      Adds seed dispersion, seed-paired Cohen's d, paired Wilcoxon
      signed-rank tests against each subtype's own T_base arm with
      Holm-Bonferroni correction inside the subtype (family size 3), and
      a per-subtype Friedman test over all four thresholds. The Wilcoxon
      call uses SciPy's automatic route, which is exact when the paired
      differences are untied and tie-aware otherwise; this is the same
      convention recompute_stats.py uses for Tables 5-8.

Usage
-----
    python variance_pertype_tests.py [--results DIR] [--out FILE]

``--results`` defaults to ``results_aggregated/`` next to this script;
``--out`` defaults to ``pertype_variance_stats.json`` next to this script.
Reproduces every number in Section 5.3, Table 9 and its footnote.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent


def _find_results() -> Path:
    """Locate results_aggregated/ whether this script sits beside it or one
    level down (as it does in the released archive)."""
    import os
    env = os.environ.get("RESULTS_AGGREGATED")
    if env:
        return Path(env)
    for cand in (HERE / "results_aggregated",
                 HERE.parent / "results_aggregated",
                 Path("results_aggregated")):
        if (cand / "lgbm_fine" / "summary.json").exists():
            return cand
    return HERE / "results_aggregated"


SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
FILE_TYPES = ["win32", "win64", "dot_net"]
PER_TYPE_THRESHOLDS = ["0.065", "0.100", "0.150", "0.200"]
BASELINE_KEY = "0.065"
METRIC = "TPR@1.0%FPR"
SPLIT = "challenge"


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, returned in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def pitman_morgan(x, y) -> tuple[float, float, float]:
    """Paired test of H0: var(x) == var(y).  Returns (t, p, r), df = n - 2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n != len(y):
        raise ValueError("pitman_morgan requires paired samples")
    r, _ = stats.pearsonr(x + y, x - y)
    denom = max(1e-300, 1.0 - r * r)
    t = r * math.sqrt((n - 2) / denom)
    return float(t), float(2 * stats.t.sf(abs(t), n - 2)), float(r)


def cohens_d_paired(a, b) -> float:
    """Cohen's d on seed-paired differences (a - b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf")


def variance_equality(results: Path) -> dict:
    fine = json.loads((results / "lgbm_fine/summary.json").read_text())["LightGBM"]

    def vals(key: str) -> list[float]:
        return fine[key][SPLIT][METRIC]["values"]

    base = vals(BASELINE_KEY)
    sd_base = float(np.std(base, ddof=1))
    grid = [k for k in sorted(fine, key=float) if k != BASELINE_KEY]

    rows, praw = [], []
    for key in grid:
        v = vals(key)
        sd = float(np.std(v, ddof=1))
        t, p, r = pitman_morgan(base, v)
        rows.append({"T": float(key), "sd_base": sd_base, "sd_T": sd,
                     "sd_ratio": sd_base / sd, "t": t, "p_raw": p, "r": r})
        praw.append(p)
    for row, adj in zip(rows, holm(praw)):
        row["p_holm"] = adj

    sweet = [r for r in rows if 0.08 <= r["T"] <= 0.18]
    return {
        "test": "Pitman-Morgan paired variance-equality test, df = n - 2 = 8",
        "metric": f"{SPLIT} {METRIC}",
        "baseline_T": float(BASELINE_KEY),
        "correction": "Holm-Bonferroni across the 10 non-baseline fine-grid points",
        "family_size": len(rows),
        "rows": rows,
        "sweet_zone_summary": {
            "n_points": len(sweet),
            "sd_ratio_min": min(r["sd_ratio"] for r in sweet),
            "sd_ratio_max": max(r["sd_ratio"] for r in sweet),
            "max_p_holm": max(r["p_holm"] for r in sweet),
            "all_holm_significant": bool(all(r["p_holm"] < 0.05 for r in sweet)),
        },
    }


def per_file_type(results: Path) -> dict:
    pt = json.loads((results / "per_type/summary.json").read_text())
    out = {}
    for ft in FILE_TYPES:
        series = {T: pt[ft][T][SPLIT][METRIC]["values"] for T in PER_TYPE_THRESHOLDS}
        fr = stats.friedmanchisquare(*[series[T] for T in PER_TYPE_THRESHOLDS])
        base = series[BASELINE_KEY]
        rows, praw = [], []
        for T in PER_TYPE_THRESHOLDS:
            if T == BASELINE_KEY:
                continue
            v = series[T]
            # No explicit method: SciPy's "auto" route uses the exact
            # signed-rank distribution when the paired differences carry no
            # ties or zeros, and the tie-aware permutation distribution when
            # they do. Forcing method="exact" would assume untied ranks and
            # misreport the .NET T=0.20 cell. This matches recompute_stats.py.
            w = stats.wilcoxon(v, base, zero_method="wilcox")
            rows.append({
                "T": float(T),
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)),
                "delta_pp": float((np.mean(v) - np.mean(base)) * 100),
                "cohens_d": cohens_d_paired(v, base),
                "p_raw": float(w.pvalue),
                "n_seeds_favoring": int(sum(1 for a, b in zip(v, base) if a > b)),
            })
            praw.append(float(w.pvalue))
        for row, adj in zip(rows, holm(praw)):
            row["p_holm"] = adj
        out[ft] = {
            "baseline": {"T": float(BASELINE_KEY), "mean": float(np.mean(base)),
                         "sd": float(np.std(base, ddof=1))},
            "friedman_k": len(PER_TYPE_THRESHOLDS),
            "friedman_chi2": float(fr.statistic),
            "friedman_p": float(fr.pvalue),
            "correction": "Holm-Bonferroni within subtype (family size 3)",
            "rows": rows,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=_find_results(),
                    help="directory holding lgbm_fine/ and per_type/ summaries")
    ap.add_argument("--out", type=Path, default=HERE / "pertype_variance_stats.json")
    args = ap.parse_args()

    if not args.results.is_dir():
        raise SystemExit(f"results directory not found: {args.results}")

    res = {"seeds": SEEDS,
           "variance_equality_fine_grid": variance_equality(args.results),
           "per_file_type": per_file_type(args.results)}

    ve = res["variance_equality_fine_grid"]
    print(f"(A) {ve['test']}\n    metric: {ve['metric']}, "
          f"baseline T = {ve['baseline_T']}\n")
    print(f"{'T':>6} {'SD_base':>9} {'SD_T':>9} {'ratio':>8} "
          f"{'t(8)':>8} {'p_raw':>11} {'p_Holm':>11}")
    for row in ve["rows"]:
        print(f"{row['T']:6.3f} {row['sd_base'] * 100:9.3f} "
              f"{row['sd_T'] * 100:9.3f} {row['sd_ratio']:7.2f}x "
              f"{row['t']:8.3f} {row['p_raw']:11.2e} {row['p_holm']:11.2e}")
    sz = ve["sweet_zone_summary"]
    print(f"\n    sweet zone: {sz['n_points']} points, SD ratio "
          f"{sz['sd_ratio_min']:.2f}x-{sz['sd_ratio_max']:.2f}x, "
          f"max p_Holm = {sz['max_p_holm']:.2e}")

    print(f"\n\n(B) per-file-type, {SPLIT} {METRIC}, n = 10 seeds\n")
    for ft, blk in res["per_file_type"].items():
        print(f"  {ft}: Friedman chi2({blk['friedman_k'] - 1}) = "
              f"{blk['friedman_chi2']:.2f}, p = {blk['friedman_p']:.2e}; "
              f"baseline {blk['baseline']['mean'] * 100:.2f} "
              f"+- {blk['baseline']['sd'] * 100:.2f}")
        for row in blk["rows"]:
            print(f"      T={row['T']:.3f}  {row['mean'] * 100:6.2f} "
                  f"+- {row['sd'] * 100:4.2f}  {row['delta_pp']:+6.2f}pp  "
                  f"d={row['cohens_d']:+6.2f}  p_Holm={row['p_holm']:.4f}  "
                  f"{row['n_seeds_favoring']}/10")

    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
