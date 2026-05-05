#!/usr/bin/env python3
"""analysis.py — Aggregate per-run JSON into summary tables and stats.

Reads results/<experiment>/per_run/*.json produced by each runner and
builds:
    results/<experiment>/summary.json         (mean/std/ci/values per group)
    results/<experiment>/stats.json           (Wilcoxon vs baseline T=0.065)
    results/tables/<name>.csv                 (publication-ready CSV tables)

Call once after all runners have finished. Fail-safe: missing runs are
noted but do not crash the aggregation.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (PATHS, BASELINE_T, PRIMARY_METRIC, PRIMARY_SPLIT, SEEDS,
                    RUN_TAG)
from metrics import aggregate
from stats import (wilcoxon_cohens, friedman_across_thresholds,
                   compare_each_to_baseline, bonferroni, holm_bonferroni)
from runners._common import setup_logger, write_json_atomic, read_json


def load_runs(per_run_dir: Path) -> list[dict]:
    if not per_run_dir.exists():
        return []
    runs = []
    for p in sorted(per_run_dir.glob("*.json")):
        try:
            runs.append(read_json(p))
        except Exception as e:
            print(f"[WARN] failed to read {p.name}: {e}", file=sys.stderr)
    return runs


# ── Cross-classifier / LGBM fine-grid aggregator ─────────────────────────────
def _aggregate_reliability_bins(runs: list, split_cal_key: str) -> list:
    """Average reliability-diagram bins across seeds (mean_pred, frac_pos, n).

    Each run's calibration["bins"] is list of {bin, lo, hi, n, mean_pred, frac_pos, gap}.
    Averaged bin emits {bin, lo, hi, n_total, mean_pred_mean, frac_pos_mean, gap_mean}.
    """
    if not runs or split_cal_key not in runs[0]:
        return []
    n_bins = len(runs[0][split_cal_key]["bins"])
    out = []
    for b in range(n_bins):
        ns, mps, fps, gaps = [], [], [], []
        for r in runs:
            bin_data = r[split_cal_key]["bins"][b]
            if bin_data["n"] > 0 and not np.isnan(bin_data["mean_pred"]):
                ns.append(bin_data["n"])
                mps.append(bin_data["mean_pred"])
                fps.append(bin_data["frac_pos"])
                gaps.append(bin_data["gap"])
        if ns:
            out.append({
                "bin": b,
                "lo": runs[0][split_cal_key]["bins"][b]["lo"],
                "hi": runs[0][split_cal_key]["bins"][b]["hi"],
                "n_total": int(sum(ns)),
                "mean_pred_avg": float(np.mean(mps)),
                "frac_pos_avg": float(np.mean(fps)),
                "gap_avg": float(np.mean(gaps)),
            })
        else:
            out.append({"bin": b, "n_total": 0})
    return out


def aggregate_threshold_sweep(per_run_dir: Path, splits=("test", "challenge"),
                              metrics_of_interest=("TPR@0.1%FPR", "TPR@1.0%FPR",
                                                   "TPR@5.0%FPR",
                                                   "ROC-AUC", "PR-AUC"),
                              cal_metrics=("ECE", "MCE", "Brier")) -> dict:
    """Group runs by (classifier, T) → aggregate over seeds."""
    runs = load_runs(per_run_dir)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["classifier"], float(r["T"]))].append(r)

    summary = {}
    for (clf, T), rs in groups.items():
        rs.sort(key=lambda r: r["seed"])
        bucket = {
            "classifier": clf,
            "T": T,
            "seeds": [r["seed"] for r in rs],
            "n_runs": len(rs),
            "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rs])),
        }
        for sp in splits:
            bucket[sp] = {}
            for m in metrics_of_interest:
                vals = [r[sp].get(m) for r in rs if m in r.get(sp, {})]
                if vals:
                    bucket[sp][m] = aggregate(vals)
            # Calibration metrics (added 2026-04-25)
            cal_key = f"{sp}_calibration"
            for cm in cal_metrics:
                vals = [r[cal_key].get(cm) for r in rs if cal_key in r and cm in r.get(cal_key, {})]
                if vals:
                    bucket[sp][cm] = aggregate(vals)
            # Reliability diagram bins averaged across seeds (for figure)
            if any(cal_key in r for r in rs):
                bucket[sp]["reliability_bins"] = _aggregate_reliability_bins(rs, cal_key)
        summary.setdefault(clf, {})[f"{T:.3f}"] = bucket
    return summary


def stats_threshold_sweep(summary: dict, baseline_T: float = BASELINE_T,
                          split: str = PRIMARY_SPLIT,
                          metric: str = PRIMARY_METRIC,
                          correction: str = "bonferroni") -> dict:
    """For each classifier, run Wilcoxon baseline vs each other T."""
    out = {"baseline_T": baseline_T, "split": split, "metric": metric,
           "correction": correction, "per_classifier": {}}
    for clf, per_T in summary.items():
        values_by_T = {}
        for T_str, bucket in per_T.items():
            T = float(T_str)
            if split in bucket and metric in bucket[split]:
                values_by_T[T] = bucket[split][metric]["values"]
        if baseline_T not in values_by_T or len(values_by_T) < 2:
            out["per_classifier"][clf] = {"skipped_reason":
                "insufficient data for Wilcoxon vs baseline"}
            continue
        # Friedman across all thresholds
        Ts = sorted(values_by_T.keys())
        mat = np.array([values_by_T[T] for T in Ts])
        friedman = friedman_across_thresholds(mat)
        pairwise = compare_each_to_baseline(values_by_T, baseline_T, correction)
        out["per_classifier"][clf] = {
            "friedman": friedman,
            "pairwise": pairwise,
            "thresholds": Ts,
        }
    return out


def aggregate_weighting(per_run_dir: Path) -> dict:
    runs = load_runs(per_run_dir)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["classifier"], r["strategy"])].append(r)
    out = {}
    for (clf, strategy), rs in groups.items():
        rs.sort(key=lambda r: r["seed"])
        bucket = {
            "classifier": clf, "strategy": strategy,
            "n_runs": len(rs),
            "seeds": [r["seed"] for r in rs],
            "weight_stats_mean": {
                k: float(np.mean([r["weight_stats"][k] for r in rs]))
                for k in ("mean", "min", "max")
            } if rs and "weight_stats" in rs[0] else {},
            "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rs])),
        }
        for sp in ("test", "challenge"):
            bucket[sp] = {}
            for m in ("TPR@0.1%FPR", "TPR@1.0%FPR", "ROC-AUC", "PR-AUC"):
                vals = [r[sp].get(m) for r in rs if m in r.get(sp, {})]
                if vals:
                    bucket[sp][m] = aggregate(vals)
        out.setdefault(clf, {})[strategy] = bucket
    return out


def stats_weighting(summary: dict, split: str = PRIMARY_SPLIT,
                    metric: str = PRIMARY_METRIC,
                    baseline_strategy: str = "uniform") -> dict:
    out = {"baseline_strategy": baseline_strategy,
           "split": split, "metric": metric, "per_classifier": {}}
    for clf, per_strategy in summary.items():
        if baseline_strategy not in per_strategy:
            out["per_classifier"][clf] = {"skipped": "no uniform baseline"}
            continue
        base_vals = per_strategy[baseline_strategy][split][metric]["values"]
        results = {}
        p_vals, keys = [], []
        for s, bucket in per_strategy.items():
            if s == baseline_strategy:
                continue
            if split not in bucket or metric not in bucket[split]:
                continue
            vals = bucket[split][metric]["values"]
            res = wilcoxon_cohens(base_vals, vals)
            results[s] = res
            p_vals.append(res["p_value"])
            keys.append(s)
        sig = holm_bonferroni(p_vals)
        for k, s_ in zip(keys, sig):
            results[k]["significant_holm"] = bool(s_)
        # Friedman across all strategies (including baseline)
        strats = [baseline_strategy] + keys
        mat = np.array([per_strategy[s][split][metric]["values"] for s in strats])
        friedman = friedman_across_thresholds(mat)
        out["per_classifier"][clf] = {"friedman": friedman, "pairwise": results}
    return out


def aggregate_per_type(per_run_dir: Path) -> dict:
    runs = load_runs(per_run_dir)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["file_type"], float(r["T"]))].append(r)
    out = {}
    for (ft, T), rs in groups.items():
        rs.sort(key=lambda r: r["seed"])
        bucket = {
            "file_type": ft, "T": T,
            "n_runs": len(rs),
            "seeds": [r["seed"] for r in rs],
            "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rs])),
        }
        for sp in ("test", "challenge"):
            bucket[sp] = {}
            for m in ("TPR@0.1%FPR", "TPR@1.0%FPR", "ROC-AUC", "PR-AUC"):
                vals = [r[sp].get(m) for r in rs if m in r.get(sp, {})]
                if vals:
                    bucket[sp][m] = aggregate(vals)
        out.setdefault(ft, {})[f"{T:.3f}"] = bucket
    return out


# ── CSV table writers ────────────────────────────────────────────────────────
def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_cross_classifier_table(summary: dict, tables_dir: Path):
    """One CSV per (split, metric)."""
    for split in ("test", "challenge"):
        for metric in ("TPR@0.1%FPR", "TPR@1.0%FPR", "ROC-AUC", "PR-AUC"):
            rows = []
            for clf, per_T in summary.items():
                for T_str, bucket in sorted(per_T.items(), key=lambda x: float(x[0])):
                    cell = bucket.get(split, {}).get(metric)
                    if cell is None:
                        continue
                    rows.append([
                        clf, float(T_str),
                        f"{cell['mean']*100:.2f}" if metric.startswith("TPR") else f"{cell['mean']:.4f}",
                        f"{cell['std']*100:.2f}"  if metric.startswith("TPR") else f"{cell['std']:.4f}",
                        cell["n"],
                    ])
            metric_slug = metric.replace("/", "_").replace("%", "pct").replace(".", "_")
            _write_csv(
                tables_dir / f"cross_classifier__{split}__{metric_slug}.csv",
                ["classifier", "T", "mean", "std", "n"], rows,
            )


def write_weighting_table(summary: dict, tables_dir: Path):
    rows = []
    for clf, per_s in summary.items():
        for s, bucket in per_s.items():
            c = bucket.get("challenge", {}).get(PRIMARY_METRIC)
            if c is None:
                continue
            rows.append([
                clf, s,
                f"{c['mean']*100:.2f}",
                f"{c['std']*100:.2f}",
                c["n"],
            ])
    _write_csv(tables_dir / "weighting__challenge__TPR1pct.csv",
               ["classifier", "strategy", "mean_pct", "std_pct", "n"], rows)


def write_per_type_table(summary: dict, tables_dir: Path):
    rows = []
    for ft, per_T in summary.items():
        for T_str, bucket in sorted(per_T.items(), key=lambda x: float(x[0])):
            c = bucket.get("challenge", {}).get(PRIMARY_METRIC)
            if c is None:
                continue
            rows.append([
                ft, float(T_str),
                f"{c['mean']*100:.2f}",
                f"{c['std']*100:.2f}",
                c["n"],
            ])
    _write_csv(tables_dir / "per_type__challenge__TPR1pct.csv",
               ["file_type", "T", "mean_pct", "std_pct", "n"], rows)


# ── Entry point ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=PATHS.output_root)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_root)
    tables_dir = root / PATHS.tables
    logger = setup_logger("analysis", root / "analysis.log")
    logger.info(f"RUN_TAG={RUN_TAG}; root={root}")

    # Experiment 1: cross-classifier
    cc_dir = root / PATHS.cross_classifier / "per_run"
    if cc_dir.exists():
        cc_sum = aggregate_threshold_sweep(cc_dir)
        write_json_atomic(cc_sum, root / PATHS.cross_classifier / "summary.json")
        cc_stats = stats_threshold_sweep(cc_sum)
        write_json_atomic(cc_stats, root / PATHS.cross_classifier / "stats.json")
        write_cross_classifier_table(cc_sum, tables_dir)
        logger.info(f"[cross_classifier] {len(cc_sum)} clf, tables written")

    # Experiment 2: LGBM fine grid
    fg_dir = root / PATHS.lgbm_fine / "per_run"
    if fg_dir.exists():
        fg_sum = aggregate_threshold_sweep(fg_dir)
        write_json_atomic(fg_sum, root / PATHS.lgbm_fine / "summary.json")
        fg_stats = stats_threshold_sweep(fg_sum)
        write_json_atomic(fg_stats, root / PATHS.lgbm_fine / "stats.json")
        logger.info(f"[lgbm_fine] written")

    # Experiment 3: weighting
    w_dir = root / PATHS.weighting / "per_run"
    if w_dir.exists():
        w_sum = aggregate_weighting(w_dir)
        write_json_atomic(w_sum, root / PATHS.weighting / "summary.json")
        w_stats = stats_weighting(w_sum)
        write_json_atomic(w_stats, root / PATHS.weighting / "stats.json")
        write_weighting_table(w_sum, tables_dir)
        logger.info(f"[weighting] written")

    # Experiment 4: per-type
    pt_dir = root / PATHS.per_type / "per_run"
    if pt_dir.exists():
        pt_sum = aggregate_per_type(pt_dir)
        write_json_atomic(pt_sum, root / PATHS.per_type / "summary.json")
        write_per_type_table(pt_sum, tables_dir)
        logger.info(f"[per_type] written")

    logger.info("[DONE] analysis complete")


if __name__ == "__main__":
    main()
