#!/usr/bin/env python3
"""metrics.py — ROC-AUC, PR-AUC, TPR@FPR, calibration, plus aggregation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

from config import TARGET_FPRS, CALIBRATION_BINS


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """TPR at the largest FPR <= target_fpr along the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    out = {
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
        "PR-AUC":  float(average_precision_score(y_true, y_score)),
    }
    for fpr in TARGET_FPRS:
        out[f"TPR@{fpr*100:.1f}%FPR"] = tpr_at_fpr(y_true, y_score, fpr)
    return out


def compute_calibration(y_true: np.ndarray, y_score: np.ndarray,
                        n_bins: int = CALIBRATION_BINS) -> dict:
    """Calibration metrics: ECE, MCE, Brier, reliability bins.

    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|
    MCE = max_b |acc(B_b) - conf(B_b)|
    Brier = mean((y_score - y_true)^2)
    bins = list of {n, mean_pred, frac_pos} for reliability diagram.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)

    bins = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (y_score >= lo) & (y_score < hi) if b < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        nb = int(mask.sum())
        if nb == 0:
            bins.append({"bin": b, "lo": float(lo), "hi": float(hi),
                         "n": 0, "mean_pred": float("nan"), "frac_pos": float("nan"),
                         "gap": float("nan")})
            continue
        mean_pred = float(y_score[mask].mean())
        frac_pos = float(y_true[mask].mean())
        gap = abs(mean_pred - frac_pos)
        bins.append({"bin": b, "lo": float(lo), "hi": float(hi),
                     "n": nb, "mean_pred": mean_pred, "frac_pos": frac_pos,
                     "gap": gap})
        ece += (nb / n) * gap
        if gap > mce:
            mce = gap

    return {
        "n": n,
        "n_bins": n_bins,
        "ECE": float(ece),
        "MCE": float(mce),
        "Brier": float(brier_score_loss(y_true, y_score)),
        "bins": bins,
    }


def aggregate(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean, "std": std, "ci95": ci95,
        "min": float(arr.min()), "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "n": int(len(arr)),
        "values": [float(v) for v in values],
    }
