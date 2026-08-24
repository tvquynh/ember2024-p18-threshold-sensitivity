#!/usr/bin/env python3
"""Aggregate and test the revision-2 control arms.

Reads the per-run JSON written by runners/run_revision2_controls.py and applies
the same pre-specified inferential protocol as the rest of the study: exact
seed-paired Wilcoxon signed-rank tests, Cohen's d on the paired differences, and
Holm-Bonferroni correction within each arm's own family of comparisons.

The three arms answer different questions and are therefore analysed
differently.

  matched_size      Does dropping the SAME NUMBER of training malware at random
                    reproduce the gain that dropping by the r >= T rule
                    produces?  The contrast is random-drop vs rule-drop at the
                    same T, plus random-drop vs the in-session baseline anchor
                    at T_base.  Run for BOTH metrics the paper claims a gain on:
                    challenge TPR@1%FPR (discrimination) and challenge ECE
                    (calibration).  They do not behave the same way.

  canonical         Does the sweet zone survive under the EMBER2024 authors'
                    released training configuration rather than the study's own?
                    The contrast is each T against T_base, within this arm.

  authors_pipeline  Does our pipeline reproduce the four figures the dataset
                    paper publishes for its all-PE partition, when run under
                    their configuration AND their stratified 90/10 split?
                    Reported against the published point values, and against
                    the canonical arm at T_base to isolate the split.

Usage:
    python analyze_revision2.py --results_root results --out revision2_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as ss

METRIC = "TPR@1.0%FPR"
SPLIT = "challenge"
BASELINE_T = 0.065
ALPHA = 0.05

# The four figures the EMBER2024 paper publishes for its all-PE partition
# (Tables 5 and 6 of the dataset paper).  Quoted, never recomputed.
PUBLISHED_ALL_PE = {
    ("test", "ROC-AUC"): 0.9982,
    ("test", "PR-AUC"): 0.9983,
    ("challenge", "ROC-AUC"): 0.9661,
    ("challenge", "PR-AUC"): 0.6354,
}


# Published rule-drop results, for the matched-size contrast. These come from
# the study's own aggregate and are quoted, never recomputed here.
#
# Resolved relative to this file first so the script runs unchanged from the
# released archive, which ships results_aggregated/ alongside it.
def _find_published() -> Path:
    import os
    env = os.environ.get("RESULTS_AGGREGATED")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for cand in (here / "results_aggregated",
                 here.parent / "results_aggregated",
                 Path("results_aggregated")):
        if (cand / "cross_classifier" / "summary.json").exists():
            return cand
    raise SystemExit(
        "results_aggregated/ not found. It ships at the root of this archive; "
        "run this script from the archive, or set RESULTS_AGGREGATED to its path."
    )


PUBLISHED = _find_published()


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    k = len(pvals)
    out = [0.0] * k
    running = 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        running = max(running, min(1.0, (k - rank) * pvals[idx]))
        out[idx] = running
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """b relative to a, on seed-paired values. Positive delta favours b."""
    d = b - a
    sd = d.std(ddof=1)
    if np.allclose(d, 0):
        return {"delta": 0.0, "cohens_d": 0.0, "p_raw": 1.0, "n": int(d.size)}
    _, p = ss.wilcoxon(a, b, alternative="two-sided")
    return {"delta": float(d.mean()),
            "cohens_d": float(d.mean() / sd) if sd > 0 else 0.0,
            "p_raw": float(p), "n": int(d.size)}


# ── metric access ─────────────────────────────────────────────────────────
# Two storage layouts have to be read with one interface: a per-run JSON keeps
# calibration under "<split>_calibration", while the pooled aggregate keeps it
# inside "<split>" next to the discrimination metrics.

METRICS = {
    # name: (per-run accessor, aggregate key, scale, higher_is_better)
    "TPR@1.0%FPR": (lambda r: r[SPLIT][METRIC], METRIC, 100.0, True),
    "ECE": (lambda r: r[f"{SPLIT}_calibration"]["ECE"], "ECE", 1.0, False),
}


def load_arm(per_run: Path, metric: str = METRIC) -> dict:
    """{T: {seed: value}} for one metric, scaled as the paper reports it."""
    get, _, scale, _ = METRICS[metric]
    out: dict[float, dict[int, float]] = {}
    for f in sorted(per_run.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        out.setdefault(float(r["T"]), {})[int(r["seed"])] = float(get(r)) * scale
    return out


def series(arm: dict, T: float, seeds: list[int]) -> np.ndarray:
    return np.array([arm[T][s] for s in seeds])


def published_by_seed(stage: str, T: float,
                      metric: str = METRIC) -> dict[int, float] | None:
    """Published per-seed values keyed by seed, never by array position.

    The aggregate records the seed list alongside the value array, so pairing
    is done on the seed itself; a positional pairing would silently mis-align
    if either side ever ran a different seed order or a partial set.
    """
    f = PUBLISHED / stage / "summary.json"
    if not f.exists():
        return None
    _, agg_key, scale, _ = METRICS[metric]
    d = json.loads(f.read_text(encoding="utf-8")).get("LightGBM", {})
    for key in (f"{T:.3f}", f"{T:g}"):
        if key in d:
            block = d[key]
            if agg_key not in block[SPLIT]:
                return None
            vals = block[SPLIT][agg_key]["values"]
            seeds = block.get("seeds")
            if seeds is None or len(seeds) != len(vals):
                return None
            return {int(s): float(v) * scale for s, v in zip(seeds, vals)}
    return None


def published_series(stage: str, T: float, seeds: list[int],
                     metric: str = METRIC) -> np.ndarray | None:
    """Published values for exactly `seeds`, in that order, or None."""
    by_seed = published_by_seed(stage, T, metric)
    if by_seed is None or not set(seeds).issubset(by_seed):
        return None
    return np.array([by_seed[s] for s in seeds])


def report_matched(arm: dict, log: list, metric: str = METRIC,
                   unit: str = "pp") -> dict:
    """Random-drop vs rule-drop and vs the in-session anchor, for one metric.

    Sign convention is fixed to "positive = better" for both metrics, so a
    lower-is-better metric such as ECE is negated before the paired tests.
    """
    _, _, _, higher_better = METRICS[metric]
    sgn = 1.0 if higher_better else -1.0

    Ts = sorted(arm)
    if BASELINE_T not in Ts:
        log.append("  WARNING: no in-session anchor at T_base; contrasts fall "
                   "back to the published baseline")
    seeds = sorted(set.intersection(*(set(arm[T]) for T in Ts)))
    log.append(f"  metric: {metric} ({'higher' if higher_better else 'lower'}"
               f" is better)")
    log.append(f"  seeds common to every threshold: {seeds}")

    anchor = series(arm, BASELINE_T, seeds) if BASELINE_T in arm else None
    if anchor is not None:
        pub = published_series("cross_classifier", BASELINE_T, seeds, metric)
        log.append(f"  in-session anchor  T={BASELINE_T}: "
                   f"{anchor.mean():.6g} +- {anchor.std(ddof=1):.3g}")
        if pub is not None:
            log.append(f"  published baseline T={BASELINE_T}: "
                       f"{pub.mean():.6g} +- {pub.std(ddof=1):.3g}  "
                       f"(parity gap {anchor.mean() - pub.mean():+.6f})")

    targets = [T for T in Ts if T != BASELINE_T]
    rows, raw_vs_rule, raw_vs_anchor = [], [], []
    for T in targets:
        rand = series(arm, T, seeds)
        rule = published_series("lgbm_fine", T, seeds, metric)
        if rule is None:
            rule = published_series("cross_classifier", T, seeds, metric)
        row = {"T": T, "random_mean": float(rand.mean()),
               "random_std": float(rand.std(ddof=1))}
        if rule is not None and rule.size == rand.size:
            r = paired(sgn * rand, sgn * rule)   # positive => rule-drop better
            row["rule_mean"] = float(rule.mean())
            row["rule_std"] = float(rule.std(ddof=1))
            row["rule_minus_random"] = float(rule.mean() - rand.mean())
            row["rule_advantage"] = r["delta"]   # sign-corrected
            row["d_rule_vs_random"] = r["cohens_d"]
            raw_vs_rule.append(r["p_raw"])
        if anchor is not None:
            a = paired(sgn * anchor, sgn * rand)  # positive => random better
            row["random_minus_anchor"] = float(rand.mean() - anchor.mean())
            row["random_advantage"] = a["delta"]
            row["d_random_vs_anchor"] = a["cohens_d"]
            row["random_rel_change_pct"] = (
                float((rand.mean() - anchor.mean()) / anchor.mean() * 100)
                if anchor.mean() else None)
            raw_vs_anchor.append(a["p_raw"])
        if "rule_mean" in row and anchor is not None and anchor.mean():
            row["rule_rel_change_pct"] = float(
                (row["rule_mean"] - anchor.mean()) / anchor.mean() * 100)
        rows.append(row)

    for key, raws in (("p_holm_rule_vs_random", raw_vs_rule),
                      ("p_holm_random_vs_anchor", raw_vs_anchor)):
        if raws:
            for row, p in zip(rows, holm(raws)):
                row[key] = p

    log.append(f"\n  {'T':>6} {'random':>18} {'rule-drop':>18} "
               f"{'rule-rand':>11} {'d':>8} {'p_holm':>8} | "
               f"{'rand-anchor':>12} {'d':>8} {'p_holm':>8}")
    for r in rows:
        log.append(
            f"  {r['T']:>6.3f} "
            f"{r['random_mean']:>9.5g}+-{r['random_std']:<8.4g} "
            f"{r.get('rule_mean', float('nan')):>9.5g}"
            f"+-{r.get('rule_std', float('nan')):<8.4g} "
            f"{r.get('rule_minus_random', float('nan')):>+11.5g} "
            f"{r.get('d_rule_vs_random', float('nan')):>+8.2f} "
            f"{r.get('p_holm_rule_vs_random', float('nan')):>8.4f} | "
            f"{r.get('random_minus_anchor', float('nan')):>+12.5g} "
            f"{r.get('d_random_vs_anchor', float('nan')):>+8.2f} "
            f"{r.get('p_holm_random_vs_anchor', float('nan')):>8.4f}")
    if not higher_better:
        log.append("  relative change vs anchor (negative = improvement):")
        for r in rows:
            log.append(f"    T={r['T']:.3f}  random {r.get('random_rel_change_pct', float('nan')):+6.1f}%"
                       f"   rule {r.get('rule_rel_change_pct', float('nan')):+6.1f}%")
    return {"metric": metric, "unit": unit, "seeds": seeds, "rows": rows,
            "anchor_mean": float(anchor.mean()) if anchor is not None else None,
            "anchor_std": float(anchor.std(ddof=1)) if anchor is not None else None}


def report_canonical(arm: dict, log: list) -> dict:
    Ts = sorted(arm)
    seeds = sorted(set.intersection(*(set(arm[T]) for T in Ts)))
    log.append(f"  seeds common to every threshold: {seeds}")
    if BASELINE_T not in arm:
        log.append("  ERROR: canonical arm has no baseline threshold")
        return {}
    base = series(arm, BASELINE_T, seeds)
    pub = published_series("cross_classifier", BASELINE_T, seeds)
    log.append(f"  canonical baseline T={BASELINE_T}: "
               f"{base.mean():.2f} +- {base.std(ddof=1):.2f}")
    if pub is not None:
        log.append(f"  study baseline     T={BASELINE_T}: "
                   f"{pub.mean():.2f} +- {pub.std(ddof=1):.2f}  "
                   f"(config gap {base.mean() - pub.mean():+.2f} pp)")

    others = [T for T in Ts if T != BASELINE_T]
    rows, raws = [], []
    for T in others:
        arm_v = series(arm, T, seeds)
        r = paired(base, arm_v)
        row = {"T": T, "mean": float(arm_v.mean()),
               "std": float(arm_v.std(ddof=1)),
               "delta": r["delta"], "cohens_d": r["cohens_d"],
               "p_raw": r["p_raw"]}
        # the study's own arm at the same T, for a like-for-like config contrast
        study = published_series("lgbm_fine", T, seeds)
        if study is None:
            study = published_series("cross_classifier", T, seeds)
        if study is not None and pub is not None:
            row["study_mean"] = float(study.mean())
            row["study_std"] = float(study.std(ddof=1))
            row["study_delta"] = float(study.mean() - pub.mean())
        rows.append(row)
        raws.append(r["p_raw"])
    for row, p in zip(rows, holm(raws)):
        row["p_holm"] = p
        row["significant"] = bool(p < ALPHA)

    chi2, pf = ss.friedmanchisquare(*[series(arm, T, seeds) for T in Ts])
    log.append(f"  Friedman chi2 = {chi2:.3f}, p = {pf:.3e}  (k = {len(Ts)})")
    log.append(f"\n  {'T':>6} {'canonical':>16} {'delta':>9} {'d':>8} "
               f"{'p_raw':>8} {'p_holm':>8} | {'study arm':>16} {'delta':>9}")
    for r in rows:
        log.append(f"  {r['T']:>6.3f} {r['mean']:>8.2f}+-{r['std']:<6.2f} "
                   f"{r['delta']:>+9.2f} {r['cohens_d']:>+8.2f} "
                   f"{r['p_raw']:>8.4f} {r['p_holm']:>8.4f}"
                   f"{' *' if r['significant'] else '  '} | "
                   f"{r.get('study_mean', float('nan')):>8.2f}"
                   f"+-{r.get('study_std', float('nan')):<6.2f} "
                   f"{r.get('study_delta', float('nan')):>+9.2f}")
    sweet = [r["T"] for r in rows if r["significant"] and r["delta"] > 0]
    log.append(f"\n  significant improvements under the released config: {sweet}")
    return {"seeds": seeds, "baseline_mean": float(base.mean()),
            "baseline_std": float(base.std(ddof=1)),
            "study_baseline_mean": float(pub.mean()) if pub is not None else None,
            "friedman_k": len(Ts),
            "friedman_chi2": float(chi2), "friedman_p": float(pf),
            "rows": rows, "sweet_zone": sweet}


def report_authors(per_run: Path, canonical_arm: dict, log: list) -> dict:
    """Fidelity of the authors' pipeline against their four published figures.

    Also isolates the 90/10 stratified split by comparing this arm with the
    canonical arm at T_base, which differs from it only in fitting on the whole
    retained pool instead of nine tenths of it.
    """
    runs = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(per_run.glob("*.json"))]
    if not runs:
        log.append("  no runs found")
        return {}
    by_seed = {int(r["seed"]): r for r in runs}
    seeds = sorted(by_seed)
    log.append(f"  {len(runs)} runs, seeds {seeds}")

    fidelity = []
    log.append(f"\n  {'split':>10} {'metric':>8} {'ours':>20} {'published':>10} "
               f"{'delta':>9} {'seeds above':>12}  verdict")
    for (split, metric), pub in PUBLISHED_ALL_PE.items():
        v = np.array([by_seed[s][split][metric] for s in seeds])
        sd = float(v.std(ddof=1))
        above = int((v > pub).sum())
        inside = bool(abs(v.mean() - pub) <= max(sd, 5e-5))
        fidelity.append({
            "split": split, "metric": metric, "published": pub,
            "mean": float(v.mean()), "std": sd,
            "min": float(v.min()), "max": float(v.max()),
            "delta": float(v.mean() - pub),
            "seeds_above_published": above,
            "within_one_sd": inside,
            "sd_distance": (abs(v.mean() - pub) / sd) if sd > 0 else 0.0,
        })
        log.append(f"  {split:>10} {metric:>8} {v.mean():>10.4f}+-{sd:<9.4f} "
                   f"{pub:>10.4f} {v.mean() - pub:>+9.4f} {above:>9}/{len(seeds)}"
                   f"  {'recovered' if inside else 'NOT recovered'}")

    out = {"seeds": seeds, "fidelity": fidelity}

    # 90/10 split vs fitting on the whole pool: same config, same seeds.
    if BASELINE_T in canonical_arm:
        common = [s for s in seeds if s in canonical_arm[BASELINE_T]]
        if len(common) >= 3:
            a = np.array([by_seed[s][SPLIT][METRIC] * 100 for s in common])
            b = series(canonical_arm, BASELINE_T, common)
            r = paired(a, b)
            out["split_effect_tpr"] = {
                "seeds": common,
                "holdout_90_10_mean": float(a.mean()),
                "full_pool_mean": float(b.mean()),
                "delta_pp": r["delta"], "cohens_d": r["cohens_d"],
                "p_raw": r["p_raw"],
            }
            log.append(f"\n  90/10 split vs full pool (challenge {METRIC}): "
                       f"{a.mean():.2f} -> {b.mean():.2f}, "
                       f"delta {r['delta']:+.2f} pp, p = {r['p_raw']:.3f}")

    # TPR@1%FPR reading convention: nearest ROC point vs largest at or below.
    diffs = []
    for s in seeds:
        r = by_seed[s]
        for split in ("test", SPLIT):
            near = r[split].get(f"{METRIC}_nearest")
            atmost = r[split].get(f"{METRIC}_atmost")
            if near is not None and atmost is not None:
                diffs.append(abs(near - atmost))
    if diffs:
        out["tpr_convention_max_abs_diff"] = float(max(diffs))
        log.append(f"  TPR convention (nearest vs at-most) max |diff| over "
                   f"{len(diffs)} comparisons: {max(diffs):.6f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out", default="revision2_stats.json")
    a = ap.parse_args()
    root = Path(a.results_root)

    log: list[str] = []
    out: dict = {"metric": METRIC, "split": SPLIT, "alpha": ALPHA}

    def banner(title: str) -> None:
        log.append("=" * 78)
        log.append(title)
        log.append("=" * 78)

    # ── matched-size arm, both metrics ────────────────────────────────────
    per_run = root / "rev2_matched" / "per_run"
    banner(f"ARM: matched_size   ({per_run})")
    if per_run.exists():
        for metric, unit in ((METRIC, "pp"), ("ECE", "abs")):
            arm = load_arm(per_run, metric)
            n = sum(len(v) for v in arm.values())
            log.append(f"\n  -- {metric} -- {n} runs over thresholds {sorted(arm)}")
            key = "matched_size" if metric == METRIC else "matched_size_ece"
            out[key] = report_matched(arm, log, metric, unit)
    else:
        log.append("  not present yet -- skipped")
    log.append("")

    # ── canonical arm ────────────────────────────────────────────────────
    per_run = root / "rev2_canonical" / "per_run"
    banner(f"ARM: canonical   ({per_run})")
    canonical_arm: dict = {}
    if per_run.exists():
        canonical_arm = load_arm(per_run)
        n = sum(len(v) for v in canonical_arm.values())
        log.append(f"  {n} runs over thresholds {sorted(canonical_arm)}")
        out["canonical"] = report_canonical(canonical_arm, log)
    else:
        log.append("  not present yet -- skipped")
    log.append("")

    # ── authors' pipeline arm ────────────────────────────────────────────
    per_run = root / "rev2_authors" / "per_run"
    banner(f"ARM: authors_pipeline   ({per_run})")
    if per_run.exists():
        out["authors_pipeline"] = report_authors(per_run, canonical_arm, log)
    else:
        log.append("  not present yet -- skipped")
    log.append("")

    text = "\n".join(log)
    print(text)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    Path(a.out).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"\nwrote {a.out} and {Path(a.out).with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
