"""Unit tests for audit_protocol.py — the corpus-agnostic Algorithm 1 module.

These tests use synthetic data only, so they run in under a second and do not
need the benchmark parquet files.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_protocol import (  # noqa: E402
    filter_by_threshold,
    k_lookup_table,
    k_to_threshold,
    sweet_zone_decision,
    threshold_sweep,
    threshold_to_k,
    weighting_falsification,
)


# ── Panel-size conversion ───────────────────────────────────────────────────
def test_threshold_to_k_matches_paper_lookup():
    """The T-to-k table printed in the discussion section."""
    expected = {
        65: {0.10: 7, 0.12: 8, 0.15: 10},
        77: {0.10: 8, 0.12: 10, 0.15: 12},
        90: {0.10: 9, 0.12: 11, 0.15: 14},
        100: {0.10: 10, 0.12: 12, 0.15: 15},
    }
    assert k_lookup_table([0.10, 0.12, 0.15], [65, 77, 90, 100]) == expected


def test_threshold_to_k_is_a_ceiling():
    assert threshold_to_k(0.5, 10) == 5     # exact hit must not round up
    assert threshold_to_k(0.51, 10) == 6


def test_baseline_ratio_is_a_rounded_representation_of_k_equals_5():
    """T_base = 0.065 is 5/77 rounded UP to three decimals.

    Feeding the rounded ratio back through the ceiling therefore yields 6,
    not 5 — which is exactly why the paper states deployment guidance in k
    rather than in the ratio. The exact ratio round-trips correctly.
    """
    assert threshold_to_k(5 / 77, 77) == 5
    assert 5 / 77 == pytest.approx(0.0649, abs=1e-4)
    assert threshold_to_k(0.065, 77) == 6


def test_k_roundtrip():
    for N in (65, 77, 100):
        for k in range(1, N):
            assert threshold_to_k(k_to_threshold(k, N), N) == k


def test_panel_size_must_be_positive():
    with pytest.raises(ValueError):
        threshold_to_k(0.12, 0)


# ── Filtering ───────────────────────────────────────────────────────────────
def test_filter_keeps_all_benign_regardless_of_r():
    y = np.array([0, 0, 1, 1])
    r = np.array([0.0, 0.9, 0.01, 0.9])
    assert filter_by_threshold(r, y, T=0.5).tolist() == [True, True, False, True]


# ── Sweep ───────────────────────────────────────────────────────────────────
def test_threshold_sweep_shapes_and_pairing():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    r = rng.random(200)
    seeds = [1, 2, 3]
    Ts = [0.1, 0.3, 0.6]

    def fake_train(keep, T, seed):
        return {"metric": float(keep.sum()) + seed}

    out = threshold_sweep(r, y, Ts, seeds, fake_train)
    assert set(out) == {"metric"}
    assert sorted(out["metric"]) == Ts
    assert all(len(v) == len(seeds) for v in out["metric"].values())


# ── Sweet-zone decision ─────────────────────────────────────────────────────
def _synthetic(better_at, n=10, seed=0):
    """Baseline at T=0.065; `better_at` gets a large, consistent lift."""
    rng = np.random.default_rng(seed)
    base = 60 + rng.normal(0, 0.2, n)
    vals = {0.065: base}
    for T in (0.10, 0.20, 0.40):
        shift = 5.0 if T == better_at else -5.0
        vals[T] = base + shift + rng.normal(0, 0.05, n)
    return vals


def test_sweet_zone_partitions_thresholds():
    dec = sweet_zone_decision(_synthetic(better_at=0.10), baseline_T=0.065)
    assert dec.sweet_zone == [0.10]
    assert sorted(dec.degradation) == [0.20, 0.40]
    assert dec.boundary == []
    assert dec.family_size == 3


def test_higher_is_better_flag_inverts_the_partition():
    vals = _synthetic(better_at=0.10)
    dec = sweet_zone_decision(vals, baseline_T=0.065, higher_is_better=False)
    assert dec.degradation == [0.10]
    assert sorted(dec.sweet_zone) == [0.20, 0.40]


def test_holm_floor_is_reported():
    """Exact Wilcoxon at n seeds cannot produce p below F * 2 / 2**n."""
    dec = sweet_zone_decision(_synthetic(better_at=0.10), baseline_T=0.065)
    for row in dec.rows:
        assert row["p_holm"] >= row["p_floor"] - 1e-12
        assert row["p_floor"] == pytest.approx(3 * 2 / 2 ** 10)


def test_baseline_must_be_in_grid():
    with pytest.raises(ValueError):
        sweet_zone_decision(_synthetic(better_at=0.10), baseline_T=0.999)


def test_unequal_seed_counts_rejected():
    vals = _synthetic(better_at=0.10)
    vals[0.20] = vals[0.20][:5]
    with pytest.raises(ValueError):
        sweet_zone_decision(vals, baseline_T=0.065)


def test_cohens_d_sign_follows_the_arm_not_the_baseline():
    """A threshold that improves on the baseline must get a positive d."""
    dec = sweet_zone_decision(_synthetic(better_at=0.10), baseline_T=0.065)
    by_T = {r["T"]: r for r in dec.rows}
    assert by_T[0.10]["cohens_d"] > 0 and by_T[0.10]["delta"] > 0
    assert by_T[0.20]["cohens_d"] < 0 and by_T[0.20]["delta"] < 0


# ── Weighting falsification ─────────────────────────────────────────────────
def test_weighting_falsification_detects_uniform_dominance():
    rng = np.random.default_rng(1)
    ref = 60 + rng.normal(0, 0.2, 10)
    vals = {"uniform": ref,
            "linear": ref - 7.5 + rng.normal(0, 0.05, 10),
            "sqrt": ref - 2.9 + rng.normal(0, 0.05, 10)}
    rep = weighting_falsification(vals, reference="uniform")
    assert rep["any_strategy_improves"] is False
    assert "falsified" in rep["verdict"]
    assert all(r["delta"] < 0 and r["significant"] for r in rep["rows"])


def test_weighting_falsification_reports_a_genuine_improvement():
    rng = np.random.default_rng(2)
    ref = 60 + rng.normal(0, 0.2, 10)
    vals = {"uniform": ref, "linear": ref + 5.0 + rng.normal(0, 0.05, 10)}
    rep = weighting_falsification(vals, reference="uniform")
    assert rep["any_strategy_improves"] is True
    assert rep["improving_strategies"] == ["linear"]


def test_missing_reference_arm_rejected():
    with pytest.raises(ValueError):
        weighting_falsification({"linear": [1, 2, 3]}, reference="uniform")
