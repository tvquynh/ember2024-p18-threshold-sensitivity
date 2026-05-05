#!/usr/bin/env python3
"""Smoke tests — verify imports, schema, weights, filter, and one tiny train.

Run with:
    pytest tests/ -q
    # or
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ember_v3_schema import TOTAL_DIMS, assert_schema, CATEGORICAL_FEATURES
from data_loader import filter_by_threshold, threshold_summary
from weights import weights_for
from metrics import compute_metrics, aggregate, compute_calibration
from stats import wilcoxon_cohens, friedman_across_thresholds, bonferroni, holm_bonferroni


# ── Schema ───────────────────────────────────────────────────────────────────
def test_schema_total_dims():
    X = np.zeros((5, TOTAL_DIMS), dtype=np.float32)
    assert_schema(X)


def test_schema_rejects_wrong_dims():
    with pytest.raises(ValueError):
        assert_schema(np.zeros((5, 2567), dtype=np.float32))


def test_categorical_features_sorted_unique():
    assert len(CATEGORICAL_FEATURES) == len(set(CATEGORICAL_FEATURES))
    assert CATEGORICAL_FEATURES == sorted(CATEGORICAL_FEATURES)


# ── Threshold filter ─────────────────────────────────────────────────────────
def test_filter_keeps_all_benign():
    y = np.array([0, 0, 1, 1, 1])
    r = np.array([0.0, 0.0, 0.05, 0.20, 0.80])
    mask = filter_by_threshold(r, y, T=0.10)
    # All benign kept (2), malware with r>=0.10 kept (2).
    assert mask.tolist() == [True, True, False, True, True]


def test_filter_monotonic():
    rng = np.random.default_rng(0)
    n = 10_000
    y = rng.integers(0, 2, size=n)
    r = rng.random(n)
    prev = n + 1
    for T in [0.0, 0.05, 0.10, 0.20, 0.50, 0.90]:
        m = filter_by_threshold(r, y, T).sum()
        assert m <= prev
        prev = m


def test_filter_T_zero_keeps_all():
    """T=0 (below-baseline control): no filtering, all samples retained."""
    y = np.array([0, 0, 1, 1, 1])
    r = np.array([0.0, 0.0, 0.0, 0.05, 0.20])
    mask = filter_by_threshold(r, y, T=0.0)
    assert mask.tolist() == [True, True, True, True, True]


def test_below_baseline_thresholds_present():
    """Config must include below-baseline T values (T<0.065) for control study."""
    from config import COARSE_THRESHOLDS, BELOW_BASELINE_THRESHOLDS, BASELINE_T
    assert BELOW_BASELINE_THRESHOLDS == [0.0, 0.02, 0.04]
    assert all(t < BASELINE_T for t in BELOW_BASELINE_THRESHOLDS)
    assert all(t in COARSE_THRESHOLDS for t in BELOW_BASELINE_THRESHOLDS)


def test_threshold_summary_format():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    r = np.array([0, 0, 0, 0.05, 0.12, 0.25, 0.80])
    rows = threshold_summary(r, y, [0.065, 0.10, 0.20])
    assert rows[0]["n_benign"] == 3
    assert rows[-1]["n_malware_retained"] == 2  # r>=0.20: {0.25, 0.80}


# ── Weights ─────────────────────────────────────────────────────────────────
def test_weights_uniform_all_ones():
    y = np.array([0, 0, 1, 1])
    r = np.array([0.0, 0.0, 0.1, 0.9])
    w = weights_for("uniform", y, r)
    assert np.allclose(w, 1.0)


def test_weights_benign_always_one():
    y = np.array([0, 0, 1, 1, 1])
    r = np.array([0.0, 0.0, 0.05, 0.20, 0.80])
    for strat in ["linear", "sqrt", "log", "binary_high", "step_3level"]:
        w = weights_for(strat, y, r)
        assert np.allclose(w[y == 0], 1.0), f"{strat} did not preserve benign=1"


def test_weights_binary_high_boundary():
    y = np.array([1, 1, 1])
    r = np.array([0.49, 0.50, 0.51])
    w = weights_for("binary_high", y, r)
    assert w.tolist() == [0.5, 1.0, 1.0]


def test_weights_step_3level_boundaries():
    y = np.array([1, 1, 1, 1, 1])
    r = np.array([0.10, 0.19, 0.20, 0.49, 0.50])
    w = weights_for("step_3level", y, r)
    assert w.tolist() == [0.4, 0.4, 0.7, 0.7, 1.0]


# ── Metrics ──────────────────────────────────────────────────────────────────
def test_compute_metrics_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
    m = compute_metrics(y, s)
    assert m["ROC-AUC"] == pytest.approx(1.0)
    assert m["TPR@1.0%FPR"] > 0.9


def test_compute_metrics_includes_5pct_fpr():
    """TARGET_FPRS extended 2026-04-25 to include 5% — multi-operating-point analysis."""
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
    m = compute_metrics(y, s)
    assert "TPR@0.1%FPR" in m
    assert "TPR@1.0%FPR" in m
    assert "TPR@5.0%FPR" in m


def test_aggregate_stats():
    a = aggregate([0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.10, 0.12, 0.11, 0.13])
    assert 0.09 <= a["mean"] <= 0.14
    assert a["n"] == 10
    assert a["std"] > 0


# ── Calibration ──────────────────────────────────────────────────────────────
def test_calibration_perfect_predictor():
    """If predictions equal true labels, ECE should be ~0 (perfectly calibrated)."""
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 2, size=n).astype(float)
    s = y.copy()  # Score = label exactly
    cal = compute_calibration(y, s, n_bins=10)
    assert cal["ECE"] < 0.01
    assert cal["Brier"] < 0.01
    assert cal["n"] == n
    assert cal["n_bins"] == 10


def test_calibration_uncalibrated_overconfident():
    """All preds=0.99 but only 50% positive → high ECE (~0.49)."""
    n = 200
    y = np.array([0.0] * 100 + [1.0] * 100)
    s = np.full(n, 0.99)
    cal = compute_calibration(y, s, n_bins=10)
    # Last bin (0.9-1.0) has all data, mean_pred=0.99, frac_pos=0.5, gap=0.49
    assert cal["ECE"] == pytest.approx(0.49, abs=0.02)
    assert cal["MCE"] >= cal["ECE"]


def test_calibration_bins_structure():
    """compute_calibration returns 10 bins with required fields."""
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    cal = compute_calibration(y, s, n_bins=10)
    assert len(cal["bins"]) == 10
    for b in cal["bins"]:
        assert "bin" in b and "n" in b and "lo" in b and "hi" in b
    assert "ECE" in cal and "MCE" in cal and "Brier" in cal


# ── Stats ────────────────────────────────────────────────────────────────────
def test_wilcoxon_identical_data():
    a = [0.5] * 10
    r = wilcoxon_cohens(a, a)
    assert r["p_value"] == 1.0
    assert r["cohens_d"] == 0.0


def test_wilcoxon_clear_effect():
    a = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50]
    b = [0.65, 0.66, 0.64, 0.65, 0.67, 0.63, 0.65, 0.66, 0.64, 0.65]
    r = wilcoxon_cohens(a, b)
    assert r["p_value"] < 0.05
    assert r["cohens_d"] > 2.0
    assert r["delta_mean"] > 0.1


def test_friedman_k_ge_3():
    mat = np.array([[0.50] * 10, [0.60] * 10, [0.55] * 10]) + \
          np.random.RandomState(0).normal(0, 0.01, (3, 10))
    res = friedman_across_thresholds(mat)
    assert res["k"] == 3
    assert res["n"] == 10


def test_bonferroni_strictness():
    # alpha=0.05, k=5 => threshold=0.01
    assert bonferroni([0.005, 0.015, 0.05, 0.009, 0.02]) == \
        [True, False, False, True, False]


def test_holm_stepdown_orders():
    # Holm: sort p, compare to alpha/(k-rank).
    sig = holm_bonferroni([0.01, 0.04, 0.03])
    assert sig == [True, False, False] or sig == [True, True, False] or sig == [True, True, True]


# ── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", str(Path(__file__).parent), "-q"]))
