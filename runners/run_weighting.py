#!/usr/bin/env python3
"""Experiment 3 — Detection-weighted training.

LightGBM × 6 strategies × 10 seeds + XGBoost × 4 strategies × 10 seeds
= 60 + 40 = 100 training runs. Manuscript Table 4 + Section 5.4.

Training set is the FULL dataset (T = baseline), not threshold-filtered.
Weights come from weights.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (WEIGHTING_STRATEGIES, WEIGHTING_STRATEGIES_XGB, PE_FILE_TYPES,
                    PARQUET_DIR_DEFAULT, SEEDS, PROTOTYPE_TRAIN_SIZE,
                    PROTOTYPE_TEST_SIZE, PROTOTYPE_SEEDS, PATHS, BASELINE_T,
                    RUN_TAG)
from data_loader import load_all, filter_by_threshold
from classifiers import train, predict_proba
from weights import weights_for
from metrics import compute_metrics
from runners._common import (setup_logger, write_json_atomic, already_done,
                             time_block)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--output_dir", default=f"{PATHS.output_root}/{PATHS.weighting}")
    p.add_argument("--lgbm_strategies", nargs="*", default=WEIGHTING_STRATEGIES)
    p.add_argument("--xgb_strategies",  nargs="*", default=WEIGHTING_STRATEGIES_XGB)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--prototype", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("weighting", out / "run.log")
    logger.info(f"RUN_TAG={RUN_TAG}; args={vars(args)}")

    seeds = PROTOTYPE_SEEDS if args.prototype else args.seeds
    proto = (PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE) if args.prototype else None

    with time_block(logger, "data load"):
        bundle = load_all(args.parquet_dir, PE_FILE_TYPES, logger, prototype_sizes=proto)

    # Baseline training set = T=0.065 (matches the "uniform" row from Exp 1).
    mask = filter_by_threshold(bundle.r_train, bundle.y_train, BASELINE_T)
    X, y, r = bundle.X_train[mask], bundle.y_train[mask], bundle.r_train[mask]
    logger.info(f"Training set @ T={BASELINE_T}: n={mask.sum():,} "
                f"(mal={(y==1).sum():,} ben={(y==0).sum():,})")

    plan = (
        [("LightGBM", s) for s in args.lgbm_strategies] +
        [("XGBoost",  s) for s in args.xgb_strategies]
    )
    total = len(plan) * len(seeds)
    logger.info(f"Plan: {total} runs")

    done = 0
    for clf_name, strategy in plan:
        w = weights_for(strategy, y, r)
        logger.info(f"--- {clf_name} / {strategy}: "
                    f"w.mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f} ---")
        for seed in seeds:
            fname = f"{clf_name}__{strategy}__seed{seed}.json"
            result_file = out / "per_run" / fname
            if args.resume and already_done(result_file):
                done += 1
                logger.info(f"[SKIP] {fname} ({done}/{total})")
                continue
            try:
                t0 = time.time()
                model = train(clf_name, X, y, seed=seed, sample_weight=w)
                tt = time.time() - t0
                p_te = predict_proba(clf_name, model, bundle.X_test)
                p_ch = predict_proba(clf_name, model, bundle.X_ch_mixed)
                res = {
                    "classifier": clf_name,
                    "strategy": strategy,
                    "seed": seed,
                    "T_baseline": BASELINE_T,
                    "n_train": int(mask.sum()),
                    "train_time_s": round(tt, 2),
                    "weight_stats": {
                        "mean": float(w.mean()),
                        "min": float(w.min()),
                        "max": float(w.max()),
                    },
                    "test": compute_metrics(bundle.y_test, p_te),
                    "challenge": compute_metrics(bundle.y_ch_mixed, p_ch),
                }
                write_json_atomic(res, result_file)
                done += 1
                logger.info(
                    f"[DONE {done}/{total}] {clf_name} {strategy} seed={seed} "
                    f"ch_TPR@1%={res['challenge']['TPR@1.0%FPR']*100:.2f}% "
                    f"({tt:.1f}s)"
                )
            except Exception as e:
                logger.exception(f"[FAIL] {clf_name} {strategy} seed={seed}: {e}")

    logger.info(f"[DONE] {done}/{total}")


if __name__ == "__main__":
    main()
