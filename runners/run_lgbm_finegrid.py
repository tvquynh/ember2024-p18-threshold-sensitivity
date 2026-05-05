#!/usr/bin/env python3
"""Experiment 2 — LightGBM fine-grained threshold sweep.

LightGBM × 11 thresholds × 10 seeds = 110 training runs.
Maps the sweet-zone boundary in Figure 2 and Table 4 (variance).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (FINE_THRESHOLDS, PARQUET_DIR_DEFAULT, PE_FILE_TYPES, SEEDS,
                    PATHS, PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE,
                    PROTOTYPE_SEEDS, RUN_TAG)
from data_loader import load_all, filter_by_threshold
from classifiers import train, predict_proba
from metrics import compute_metrics, compute_calibration
from runners._common import (setup_logger, write_json_atomic, run_name,
                             already_done, time_block)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--output_dir", default=f"{PATHS.output_root}/{PATHS.lgbm_fine}")
    p.add_argument("--thresholds", nargs="*", type=float, default=FINE_THRESHOLDS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--prototype", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("lgbm_fine", out / "run.log")
    logger.info(f"RUN_TAG={RUN_TAG}; args={vars(args)}")

    seeds = PROTOTYPE_SEEDS if args.prototype else args.seeds
    proto = (PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE) if args.prototype else None

    with time_block(logger, "data load"):
        bundle = load_all(args.parquet_dir, PE_FILE_TYPES, logger, prototype_sizes=proto)

    total = len(args.thresholds) * len(seeds)
    logger.info(f"Plan: LightGBM × {len(args.thresholds)} T × {len(seeds)} seeds = {total}")

    done = 0
    for T in args.thresholds:
        mask = filter_by_threshold(bundle.r_train, bundle.y_train, T)
        X_T, y_T = bundle.X_train[mask], bundle.y_train[mask]
        n_mal = int((y_T == 1).sum())
        n_ben = int((y_T == 0).sum())
        logger.info(f"=== T={T:.3f} | mal={n_mal:,} ben={n_ben:,} ===")

        for seed in seeds:
            result_file = out / "per_run" / f"{run_name('LightGBM', T, seed)}.json"
            if args.resume and already_done(result_file):
                done += 1
                logger.info(f"[SKIP] {result_file.name} ({done}/{total})")
                continue
            try:
                t0 = time.time()
                model = train("LightGBM", X_T, y_T, seed=seed)
                tt = time.time() - t0
                p_te = predict_proba("LightGBM", model, bundle.X_test)
                p_ch = predict_proba("LightGBM", model, bundle.X_ch_mixed)
                res = {
                    "classifier": "LightGBM",
                    "T": T, "seed": seed,
                    "n_train": int(mask.sum()),
                    "train_time_s": round(tt, 2),
                    "test": compute_metrics(bundle.y_test, p_te),
                    "test_calibration": compute_calibration(bundle.y_test, p_te),
                    "challenge": compute_metrics(bundle.y_ch_mixed, p_ch),
                    "challenge_calibration": compute_calibration(bundle.y_ch_mixed, p_ch),
                }
                write_json_atomic(res, result_file)
                done += 1
                logger.info(
                    f"[DONE {done}/{total}] T={T:.3f} seed={seed} "
                    f"ch_TPR@1%={res['challenge']['TPR@1.0%FPR']*100:.2f}% "
                    f"({tt:.1f}s)"
                )
            except Exception as e:
                logger.exception(f"[FAIL] T={T} seed={seed}: {e}")

    logger.info(f"[DONE] {done}/{total}")


if __name__ == "__main__":
    main()
