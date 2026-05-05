#!/usr/bin/env python3
"""Experiment 1 — Cross-classifier threshold sensitivity (coarse grid).

4 classifiers × 6 thresholds × 10 seeds = 240 training runs.

Manuscript Table 2 + Figure 2.

Usage:
    python runners/run_cross_classifier.py \
        --parquet_dir E:/project_data/parquet_clean-week \
        --output_dir  results/cross_classifier
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (COARSE_THRESHOLDS, CLASSIFIER_NAMES, MLP_PARAMS,
                    PE_FILE_TYPES, PARQUET_DIR_DEFAULT, SEEDS, PROTOTYPE_TRAIN_SIZE,
                    PROTOTYPE_TEST_SIZE, PROTOTYPE_SEEDS, PATHS, RUN_TAG)
from data_loader import load_all, filter_by_threshold
from classifiers import train, predict_proba, stratified_val_split
from metrics import compute_metrics, compute_calibration
from runners._common import (setup_logger, write_json_atomic, run_name,
                             already_done, time_block)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--output_dir", default=f"{PATHS.output_root}/{PATHS.cross_classifier}")
    p.add_argument("--classifiers", nargs="*", default=CLASSIFIER_NAMES,
                   help=f"Subset of {CLASSIFIER_NAMES}")
    p.add_argument("--thresholds", nargs="*", type=float, default=COARSE_THRESHOLDS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--prototype", action="store_true",
                   help="Smoke-test with ~20K train, 2 seeds")
    p.add_argument("--resume", action="store_true",
                   help="Skip (clf,T,seed) whose result file already exists")
    return p.parse_args()


def run_one(clf_name: str, X_tr: np.ndarray, y_tr: np.ndarray,
            bundle, seed: int, logger) -> dict:
    X_val, y_val = None, None
    X_tr_use, y_tr_use = X_tr, y_tr
    if clf_name == "MLP":
        X_tr_use, y_tr_use, X_val, y_val = stratified_val_split(
            X_tr, y_tr, MLP_PARAMS["val_fraction"], seed
        )

    t0 = time.time()
    model = train(clf_name, X_tr_use, y_tr_use, seed=seed,
                  X_val=X_val, y_val=y_val)
    train_time = time.time() - t0

    # Standard test split (all test samples).
    p_test = predict_proba(clf_name, model, bundle.X_test)
    m_test = compute_metrics(bundle.y_test, p_test)
    cal_test = compute_calibration(bundle.y_test, p_test)

    # Challenge mixed (test benign + challenge malware).
    p_ch = predict_proba(clf_name, model, bundle.X_ch_mixed)
    m_ch = compute_metrics(bundle.y_ch_mixed, p_ch)
    cal_ch = compute_calibration(bundle.y_ch_mixed, p_ch)

    return {
        "classifier": clf_name,
        "seed": seed,
        "n_train": int(len(y_tr_use)),
        "n_val": int(len(y_val)) if y_val is not None else 0,
        "train_time_s": round(train_time, 2),
        "test": m_test,
        "test_calibration": cal_test,
        "challenge": m_ch,
        "challenge_calibration": cal_ch,
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("cross_classifier", out / "run.log")
    logger.info(f"RUN_TAG={RUN_TAG}")
    logger.info(f"args={vars(args)}")

    seeds = PROTOTYPE_SEEDS if args.prototype else args.seeds
    proto_sizes = (PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE) if args.prototype else None

    with time_block(logger, "data load"):
        bundle = load_all(args.parquet_dir, PE_FILE_TYPES, logger,
                          prototype_sizes=proto_sizes)

    total_models = len(args.classifiers) * len(args.thresholds) * len(seeds)
    logger.info(f"Plan: {len(args.classifiers)} clf × "
                f"{len(args.thresholds)} T × {len(seeds)} seeds "
                f"= {total_models} models")

    done = 0
    for T in args.thresholds:
        mask = filter_by_threshold(bundle.r_train, bundle.y_train, T)
        n_mal = int(((bundle.y_train == 1) & mask).sum())
        n_ben = int((bundle.y_train == 0).sum())
        logger.info(f"=== T={T:.3f} | train: mal={n_mal:,}  ben={n_ben:,} "
                    f"(total {mask.sum():,}) ===")
        X_T = bundle.X_train[mask]
        y_T = bundle.y_train[mask]

        for clf_name in args.classifiers:
            for seed in seeds:
                result_file = out / "per_run" / f"{run_name(clf_name, T, seed)}.json"
                if args.resume and already_done(result_file):
                    done += 1
                    logger.info(f"[SKIP] {result_file.name} exists ({done}/{total_models})")
                    continue
                try:
                    res = run_one(clf_name, X_T, y_T, bundle, seed, logger)
                    res["T"] = T
                    write_json_atomic(res, result_file)
                    done += 1
                    ch_tpr = res["challenge"].get("TPR@1.0%FPR", float("nan"))
                    logger.info(
                        f"[DONE {done}/{total_models}] {clf_name} T={T:.3f} "
                        f"seed={seed} ch_TPR@1%={ch_tpr*100:.2f}% "
                        f"({res['train_time_s']}s)"
                    )
                except Exception as e:
                    logger.exception(f"[FAIL] {clf_name} T={T} seed={seed}: {e}")

    logger.info(f"[DONE] {done}/{total_models} runs in output={out}")


if __name__ == "__main__":
    main()
