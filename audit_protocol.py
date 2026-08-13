#!/usr/bin/env python3
"""audit_protocol.py — Reference implementation of the labeling-threshold
sensitivity audit protocol (Algorithm 1 in the manuscript).

This module is deliberately **corpus-agnostic**. It knows nothing about the
benchmark used in the paper: you supply a detection-ratio vector, a training
callable, and a threshold grid, and it runs the four phases of Algorithm 1:

    1. Training sweep          -> threshold_sweep()
    2. Metric evaluation       -> (delegated to your train_fn / metrics.py)
    3. Statistical testing     -> sweet_zone_decision()
    4. Weighting falsification -> weighting_falsification()

plus the panel-size conversion used to state guidance in raw scanner counts
rather than ratios (threshold_to_k / k_to_threshold).

The point of the module is that the audit is a *protocol*, not a result: any
corpus whose malware labels come from a multi-scanner consensus vote can be
audited the same way. `run_audit()` chains the phases end-to-end.

----------------------------------------------------------------------------
Reproducing the paper's decision WITHOUT retraining
----------------------------------------------------------------------------
The released `results_aggregated/` tree already contains the per-seed metric
vectors for every (classifier, threshold) cell, so phases 3 and 4 can be
re-run in seconds:

    python audit_protocol.py --from-release results_aggregated

That prints the sweet zone, the boundary set, the degradation zone, and the
weighting falsification table, and exits non-zero if any of them disagree
with the values reported in the manuscript.
----------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from stats import holm_bonferroni, wilcoxon_cohens, friedman_across_thresholds

ALPHA_DEFAULT = 0.05


# ── Phase 0: threshold <-> scanner-count conversion ─────────────────────────
def threshold_to_k(T: float, panel_size: int) -> int:
    """Smallest scanner count k satisfying k / panel_size >= T.

    Guidance stated as a ratio silently changes meaning when the scanner
    panel grows or shrinks; stating it as k makes it panel-invariant.

    >>> threshold_to_k(0.12, 100)
    12
    >>> threshold_to_k(0.12, 77)
    10
    """
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    return int(math.ceil(T * panel_size - 1e-12))


def k_to_threshold(k: int, panel_size: int) -> float:
    """Detection ratio corresponding to k flagging scanners."""
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    return k / panel_size


def k_lookup_table(thresholds: Sequence[float],
                   panel_sizes: Sequence[int]) -> dict:
    """Panel-size lookup table: {panel_size: {T: k}}."""
    return {N: {T: threshold_to_k(T, N) for T in thresholds}
            for N in panel_sizes}


# ── Phase 1: training sweep ─────────────────────────────────────────────────
def filter_by_threshold(r: np.ndarray, y: np.ndarray, T: float) -> np.ndarray:
    """Boolean keep-mask: retain every benign sample, and malware with r >= T.

    Only the *training* labels are re-thresholded; evaluation populations are
    held fixed across the sweep, so differences are attributable to the
    training-label rule alone.
    """
    r = np.asarray(r)
    y = np.asarray(y)
    return (y == 0) | (r >= T)


def threshold_sweep(r_train: np.ndarray,
                    y_train: np.ndarray,
                    thresholds: Sequence[float],
                    seeds: Sequence[int],
                    train_fn: Callable[[np.ndarray, float, int], Mapping[str, float]],
                    progress: Callable[[str], None] | None = None) -> dict:
    """Run phase 1+2 of Algorithm 1.

    Parameters
    ----------
    r_train, y_train
        Detection ratio and binary label for every training sample. Benign
        samples are retained at every threshold regardless of their r.
    thresholds
        The grid T_1..T_K. The baseline must be a member (see
        `sweet_zone_decision`).
    seeds
        Random seeds; the same set is used at every threshold so that all
        comparisons are seed-paired.
    train_fn(keep_mask, T, seed) -> {metric_name: value}
        Caller-supplied. Trains one model on the retained subset and returns
        a flat mapping of evaluation metrics. Anything the caller returns is
        carried through; the audit itself is metric-agnostic.

    Returns
    -------
    {metric_name: {T: [value_per_seed, ...]}}
    """
    out: dict[str, dict[float, list[float]]] = {}
    for T in thresholds:
        keep = filter_by_threshold(r_train, y_train, T)
        if progress:
            progress(f"T={T:.3f}: retained {int(keep.sum()):,}/{keep.size:,}")
        for seed in seeds:
            metrics = train_fn(keep, T, seed)
            for name, value in metrics.items():
                out.setdefault(name, {}).setdefault(T, []).append(float(value))
    return out


# ── Phase 3: statistics and the sweet-zone decision ─────────────────────────
@dataclass
class SweetZoneDecision:
    baseline_T: float
    alpha: float
    family_size: int
    sweet_zone: list[float] = field(default_factory=list)
    boundary: list[float] = field(default_factory=list)
    degradation: list[float] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    friedman: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "baseline_T": self.baseline_T,
            "alpha": self.alpha,
            "family_size": self.family_size,
            "sweet_zone": self.sweet_zone,
            "boundary": self.boundary,
            "degradation": self.degradation,
            "friedman": self.friedman,
            "rows": self.rows,
        }


def sweet_zone_decision(values_by_T: Mapping[float, Sequence[float]],
                        baseline_T: float,
                        alpha: float = ALPHA_DEFAULT,
                        higher_is_better: bool = True) -> SweetZoneDecision:
    """Phase 3 of Algorithm 1: Friedman, paired Wilcoxon, Holm, partition.

    Each non-baseline threshold is compared to the baseline with an exact
    Wilcoxon signed-rank test on the seed-paired differences; the family is
    the K-1 non-baseline thresholds of this one classifier, and Holm-Bonferroni
    controls the family-wise error rate within it.

    Note on the resolution floor: with n seeds the smallest attainable exact
    two-sided p-value is 2 / 2**n, so no Holm-corrected p can fall below
    family_size * 2 / 2**n. `p_floor` records that bound so callers do not
    over-interpret a p-value that is simply pinned to the floor.
    """
    Ts = sorted(float(T) for T in values_by_T)
    if baseline_T not in Ts:
        raise ValueError(f"baseline_T={baseline_T} not in threshold grid {Ts}")
    base = np.asarray(values_by_T[baseline_T], dtype=float)
    n = base.size
    others = [T for T in Ts if T != baseline_T]

    raw, stats_rows = [], []
    for T in others:
        arm = np.asarray(values_by_T[T], dtype=float)
        if arm.size != n:
            raise ValueError(f"T={T} has {arm.size} seeds, baseline has {n}")
        res = wilcoxon_cohens(base, arm)
        raw.append(res["p_value"])
        stats_rows.append({
            "T": T,
            "mean": float(arm.mean()),
            "std": float(arm.std(ddof=1)),
            "delta": float(arm.mean() - base.mean()),
            "cohens_d": res["cohens_d"],
            "p_raw": res["p_value"],
        })

    F = len(others)
    rejected = holm_bonferroni(raw, alpha=alpha)
    order = np.argsort(raw)
    p_holm = [0.0] * F
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (F - rank) * raw[idx]))
        p_holm[idx] = running

    decision = SweetZoneDecision(
        baseline_T=baseline_T, alpha=alpha, family_size=F,
        friedman=friedman_across_thresholds(
            np.array([values_by_T[T] for T in Ts], dtype=float)),
    )
    for row, ph, rej in zip(stats_rows, p_holm, rejected):
        row["p_holm"] = ph
        row["significant"] = bool(rej)
        row["p_floor"] = F * 2.0 / (2.0 ** n)
        better = row["delta"] > 0 if higher_is_better else row["delta"] < 0
        if rej and better:
            decision.sweet_zone.append(row["T"])
        elif rej:
            decision.degradation.append(row["T"])
        else:
            decision.boundary.append(row["T"])
        decision.rows.append(row)
    return decision


# ── Phase 4: weighting falsification ────────────────────────────────────────
def weighting_falsification(values_by_strategy: Mapping[str, Sequence[float]],
                            reference: str = "uniform",
                            alpha: float = ALPHA_DEFAULT) -> dict:
    """Phase 4: can soft per-sample weights substitute for hard filtering?

    Every candidate weighting is compared to the unweighted reference arm on
    seed-paired values, Holm-corrected within the classifier's own family.
    The protocol treats this as a falsification pass: the claim "soft
    weighting can replace hard filtering" survives only if some strategy
    shows a significant positive effect.
    """
    names = [s for s in values_by_strategy if s != reference]
    if reference not in values_by_strategy:
        raise ValueError(f"reference arm {reference!r} missing")
    ref = np.asarray(values_by_strategy[reference], dtype=float)

    raw, rows = [], []
    for s in names:
        arm = np.asarray(values_by_strategy[s], dtype=float)
        res = wilcoxon_cohens(ref, arm)
        raw.append(res["p_value"])
        rows.append({"strategy": s,
                     "mean": float(arm.mean()),
                     "delta": float(arm.mean() - ref.mean()),
                     "cohens_d": res["cohens_d"],
                     "p_raw": res["p_value"]})

    F = len(names)
    rejected = holm_bonferroni(raw, alpha=alpha)
    order = np.argsort(raw)
    running = 0.0
    p_holm = [0.0] * F
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (F - rank) * raw[idx]))
        p_holm[idx] = running
    for row, ph, rej in zip(rows, p_holm, rejected):
        row["p_holm"] = ph
        row["significant"] = bool(rej)

    improved = [r["strategy"] for r in rows if r["significant"] and r["delta"] > 0]
    return {
        "reference": reference,
        "family_size": F,
        "rows": rows,
        "any_strategy_improves": bool(improved),
        "improving_strategies": improved,
        "verdict": ("soft weighting substitutes for hard filtering"
                    if improved else
                    "falsified: no weighting scheme improves on the reference"),
    }


# ── End-to-end driver ───────────────────────────────────────────────────────
def run_audit(r_train, y_train, thresholds, seeds, train_fn, baseline_T,
              primary_metric: str, alpha: float = ALPHA_DEFAULT,
              panel_sizes: Sequence[int] = (65, 77, 90, 100),
              progress: Callable[[str], None] | None = None) -> dict:
    """Chain all four phases of Algorithm 1 and return a report dict."""
    swept = threshold_sweep(r_train, y_train, thresholds, seeds, train_fn,
                            progress=progress)
    if primary_metric not in swept:
        raise KeyError(f"train_fn never returned {primary_metric!r}; "
                       f"it returned {sorted(swept)}")
    decision = sweet_zone_decision(swept[primary_metric], baseline_T, alpha)
    return {
        "primary_metric": primary_metric,
        "sweep": {m: {str(T): v for T, v in d.items()} for m, d in swept.items()},
        "decision": decision.to_dict(),
        "k_lookup": k_lookup_table(thresholds, panel_sizes),
    }


# ── Replay the published decision from the release, no retraining ───────────
def _replay(release_dir: Path, metric: str = "TPR@1.0%FPR",
            split: str = "challenge", alpha: float = ALPHA_DEFAULT) -> int:
    fine = json.loads((release_dir / "lgbm_fine" / "summary.json").read_text())
    values = {float(T): d[split][metric]["values"]
              for T, d in fine["LightGBM"].items()}
    dec = sweet_zone_decision(values, baseline_T=0.065, alpha=alpha)

    print(f"Fine grid, LightGBM, {split} {metric}, n={len(next(iter(values.values())))} seeds")
    print(f"  Friedman chi2 = {dec.friedman['stat']:.3f}")
    print(f"  family size F = {dec.family_size}, "
          f"exact-Wilcoxon Holm floor = {dec.rows[0]['p_floor']:.5f}")
    print(f"  sweet zone   : {dec.sweet_zone}")
    print(f"  boundary     : {dec.boundary}")
    print(f"  degradation  : {dec.degradation}")

    ok = True
    peak = max(dec.rows, key=lambda r: r["delta"])
    if abs(peak["T"] - 0.12) > 1e-9:
        print(f"  MISMATCH: peak at T={peak['T']}, manuscript reports T=0.12")
        ok = False
    if abs(peak["delta"] * 100 - 4.73) > 0.01:
        print(f"  MISMATCH: peak delta {peak['delta']*100:.2f}pp, manuscript reports +4.73pp")
        ok = False
    if abs(peak["cohens_d"] - 2.03) > 0.01:
        print(f"  MISMATCH: peak d={peak['cohens_d']:.2f}, manuscript reports 2.03")
        ok = False
    print(f"  peak: T={peak['T']}, delta={peak['delta']*100:+.2f}pp, "
          f"d={peak['cohens_d']:.2f}, p_holm={peak['p_holm']:.4f}")

    wt = json.loads((release_dir / "weighting" / "summary.json").read_text())
    for clf in sorted(wt):
        vals = {s: d[split][metric]["values"] for s, d in wt[clf].items()}
        rep = weighting_falsification(vals, reference="uniform", alpha=alpha)
        print(f"\n{clf} weighting audit ({rep['family_size']} strategies vs uniform)")
        for row in sorted(rep["rows"], key=lambda r: r["delta"]):
            print(f"  {row['strategy']:<14} delta={row['delta']*100:+6.2f}pp  "
                  f"d={row['cohens_d']:+6.2f}  p_holm={row['p_holm']:.4f}  "
                  f"{'sig' if row['significant'] else '--'}")
        print(f"  verdict: {rep['verdict']}")
        if rep["any_strategy_improves"]:
            print("  MISMATCH: manuscript reports that every scheme degrades detection")
            ok = False

    print("\nPanel-size lookup (k for each T):")
    table = k_lookup_table([0.10, 0.12, 0.15], [65, 77, 90, 100])
    print(f"  {'N':>4}  " + "  ".join(f"T={T}" for T in [0.10, 0.12, 0.15]))
    for N, row in table.items():
        print(f"  {N:>4}  " + "  ".join(f"k>={row[T]:<4}" for T in [0.10, 0.12, 0.15]))

    print("\n" + ("All replayed values match the manuscript." if ok
                  else "REPLAY FAILED — values disagree with the manuscript."))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--from-release", type=Path, metavar="DIR",
                    help="replay the published decision from an aggregated "
                         "results directory (no training required)")
    ap.add_argument("--metric", default="TPR@1.0%FPR")
    ap.add_argument("--split", default="challenge")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    args = ap.parse_args(argv)

    if args.from_release:
        return _replay(args.from_release, args.metric, args.split, args.alpha)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
