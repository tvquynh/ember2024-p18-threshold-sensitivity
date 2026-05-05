#!/usr/bin/env python3
"""stats.py — Wilcoxon + Cohen d + Friedman + Bonferroni/Holm.

Used by analysis.py to build the publication tables.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy import stats as scistats

from config import ALPHA


def wilcoxon_cohens(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired Wilcoxon + Cohen's d on (b - a)."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = arr_b - arr_a
    if np.all(diff == 0) or len(diff) < 2:
        return {"stat": 0.0, "p_value": 1.0, "cohens_d": 0.0,
                "delta_mean": float(np.mean(diff)), "significant": False,
                "n_pairs": int(len(diff))}
    stat, p = scistats.wilcoxon(arr_a, arr_b, alternative="two-sided")
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd > 0 else 0.0
    return {
        "stat": float(stat),
        "p_value": float(p),
        "cohens_d": d,
        "delta_mean": float(np.mean(diff)),
        "significant": bool(p < ALPHA),
        "n_pairs": int(len(diff)),
    }


def friedman_across_thresholds(matrix: np.ndarray) -> dict:
    """`matrix` shape (n_thresholds, n_seeds). Non-parametric k-sample test."""
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        return {"stat": float("nan"), "p_value": float("nan"), "k": 0, "n": 0}
    stat, p = scistats.friedmanchisquare(*[row for row in matrix])
    return {"stat": float(stat), "p_value": float(p),
            "k": int(matrix.shape[0]), "n": int(matrix.shape[1])}


def bonferroni(p_values: List[float], alpha: float = ALPHA) -> List[bool]:
    k = max(1, len(p_values))
    return [p < alpha / k for p in p_values]


def holm_bonferroni(p_values: List[float], alpha: float = ALPHA) -> List[bool]:
    k = len(p_values)
    order = np.argsort(p_values)
    sig = [False] * k
    for rank, idx in enumerate(order):
        thr = alpha / (k - rank)
        if p_values[idx] <= thr:
            sig[idx] = True
        else:
            break
    return sig


def compare_each_to_baseline(values_by_T: dict, baseline_T: float,
                             correction: str = "bonferroni") -> dict:
    """Pairwise Wilcoxon: baseline vs every other threshold. Correction across family."""
    base = values_by_T[baseline_T]
    keys, p_vals, results = [], [], {}
    for T, vals in values_by_T.items():
        if T == baseline_T:
            continue
        r = wilcoxon_cohens(base, vals)
        results[T] = r
        keys.append(T)
        p_vals.append(r["p_value"])
    if correction == "bonferroni":
        sig = bonferroni(p_vals, ALPHA)
    else:
        sig = holm_bonferroni(p_vals, ALPHA)
    for T, s in zip(keys, sig):
        results[T]["significant_corrected"] = bool(s)
    return {
        "baseline_T": baseline_T,
        "correction": correction,
        "family_size": len(keys),
        "alpha": ALPHA,
        "per_threshold": results,
    }
