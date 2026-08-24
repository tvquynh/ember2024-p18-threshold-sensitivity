#!/usr/bin/env python3
"""Why the detection-weighting damage does not follow any single summary
statistic of the weight function (Section 6.4).

For each of the five non-uniform weighting formulas of Table 2, this script
computes two candidate "aggressiveness" summaries over the retained
detection-ratio distribution at T_base, and prints them next to the measured
LightGBM degradation:

  1. suppression factor  w(1.0) / w(0.10)  -- how deeply a low-consensus
     sample is discounted relative to a fully corroborated one;
  2. weight mass removed  1 - E[w(r)]      -- how much total gradient weight
     the scheme withdraws from the retained malware pool.

Neither orders the degradations. The manuscript reports that failure rather
than selecting whichever summary happens to fit, and notes the two weaker
regularities that do hold (damage follows steepness within the smooth family;
both discrete-cutoff schemes cost more than either gentle smooth scheme).

The retained detection-ratio distribution is read from the ten-bin histogram
stored in mechanism_analysis.json, so no model and no raw data are needed.

Usage:
    python weighting_mass.py [--mechanism PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_mechanism() -> Path:
    """mechanism_analysis.json ships beside this script in the paper folder and
    in revision1_analyses/ in the released archive."""
    for cand in (HERE / "mechanism_analysis.json",
                 HERE.parent / "revision1_analyses" / "mechanism_analysis.json",
                 HERE.parent / "mechanism_analysis.json",
                 Path("mechanism_analysis.json")):
        if cand.exists():
            return cand
    return HERE / "mechanism_analysis.json"


# Table 2 of the manuscript; must stay in step with weights.py in the archive.
FORMULAS = {
    "linear": lambda r: r,
    "sqrt": lambda r: math.sqrt(r),
    "log": lambda r: math.log(1 + 10 * r) / math.log(11),
    "binary_high": lambda r: 1.0 if r >= 0.5 else 0.5,
    "step_3level": lambda r: 1.0 if r >= 0.5 else (0.7 if r >= 0.2 else 0.4),
}

# Measured LightGBM degradation vs uniform, Table 8, challenge TPR@1%FPR (pp).
LGBM_DELTA_PP = {
    "sqrt": -2.90, "log": -3.56, "step_3level": -4.50,
    "binary_high": -6.61, "linear": -7.50,
}

BASELINE_T = 0.065
PROBE_R = 0.10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mechanism", type=Path,
                    default=_find_mechanism(),
                    help="output of mechanism_analysis.py")
    args = ap.parse_args()

    diag = json.loads(args.mechanism.read_text(encoding="utf-8"))["diagnostics"]
    base = next(d for d in diag if abs(d["T"] - BASELINE_T) < 1e-9)
    hist = base["detection_ratio_histogram_10bins"]
    edges, counts = hist["edges"], hist["counts"]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    total = sum(counts)

    # share of the retained pool below the binary_high / step_3level cutoff
    def share_below(x: float) -> float:
        acc = 0.0
        for lo, hi, c in zip(edges[:-1], edges[1:], counts):
            if hi <= x:
                acc += c
            elif lo < x < hi:
                acc += c * (x - lo) / (hi - lo)
        return acc / total

    print(f"retained malware at T_base = {BASELINE_T}: {total:,}")
    print(f"share with r < 0.5: {share_below(0.5):.1%}   "
          f"share with r < 0.2: {share_below(0.2):.1%}\n")

    rows = []
    for name, f in FORMULAS.items():
        mean_w = sum(c * f(m) for c, m in zip(counts, mids)) / total
        rows.append({
            "scheme": name,
            "suppression_factor": f(1.0) / f(PROBE_R),
            "mean_weight": mean_w,
            "mass_removed": 1.0 - mean_w,
            "lgbm_delta_pp": LGBM_DELTA_PP[name],
        })

    print(f"{'scheme':13s} {'w(1)/w(0.1)':>12} {'mass removed':>13} "
          f"{'LGBM delta':>11}")
    for r in sorted(rows, key=lambda r: r["mass_removed"]):
        print(f"{r['scheme']:13s} {r['suppression_factor']:11.2f}x "
              f"{r['mass_removed']:12.1%} {r['lgbm_delta_pp']:+11.2f}")

    by_supp = [r["scheme"] for r in sorted(rows, key=lambda r: r["suppression_factor"])]
    by_mass = [r["scheme"] for r in sorted(rows, key=lambda r: r["mass_removed"])]
    by_harm = [r["scheme"] for r in sorted(rows, key=lambda r: -r["lgbm_delta_pp"])]
    print(f"\nleast to most, by suppression factor: {by_supp}")
    print(f"least to most, by mass removed      : {by_mass}")
    print(f"least to most, by measured damage   : {by_harm}")
    print(f"\nsuppression factor predicts damage: {by_supp == by_harm}")
    print(f"mass removed predicts damage      : {by_mass == by_harm}")

    smooth = ["sqrt", "log", "linear"]
    sm_by_supp = sorted(smooth, key=lambda s: next(
        r["suppression_factor"] for r in rows if r["scheme"] == s))
    sm_by_harm = sorted(smooth, key=lambda s: -LGBM_DELTA_PP[s])
    print(f"within the smooth family, steepness predicts damage: "
          f"{sm_by_supp == sm_by_harm}  ({sm_by_harm})")
    gentle = max(LGBM_DELTA_PP[s] for s in ("sqrt", "log"))
    discrete = max(LGBM_DELTA_PP[s] for s in ("binary_high", "step_3level"))
    print(f"both discrete schemes cost more than either gentle smooth scheme: "
          f"{discrete < gentle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
