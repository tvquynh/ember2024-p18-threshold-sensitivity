#!/usr/bin/env python3
"""Experiment 4 — Per-file-type LightGBM threshold sensitivity.

Per PE subtype (Win32, Win64, .NET): LightGBM × 4 thresholds × 10 seeds.
Total: 3 × 4 × 10 = 120 training runs. Manuscript Table 5.

Per-type evaluation: train on only-subtype malware + only-subtype benign
(if any); evaluate on the subtype slice of test and challenge.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (PER_TYPE_THRESHOLDS, PE_FILE_TYPES, PARQUET_DIR_DEFAULT,
                    SEEDS, PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE,
                    PROTOTYPE_SEEDS, PATHS, RUN_TAG)
from data_loader import load_all, filter_by_threshold
from classifiers import train, predict_proba
from metrics import compute_metrics
from runners._common import (setup_logger, write_json_atomic, already_done,
                             time_block)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--output_dir", default=f"{PATHS.output_root}/{PATHS.per_type}")
    p.add_argument("--file_types", nargs="*", default=PE_FILE_TYPES)
    p.add_argument("--thresholds", nargs="*", type=float, default=PER_TYPE_THRESHOLDS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--prototype", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("per_type", out / "run.log")
    logger.info(f"RUN_TAG={RUN_TAG}; args={vars(args)}")

    seeds = PROTOTYPE_SEEDS if args.prototype else args.seeds
    proto = (PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE) if args.prototype else None

    with time_block(logger, "data load"):
        bundle = load_all(args.parquet_dir, args.file_types, logger,
                          prototype_sizes=proto)

    total = len(args.file_types) * len(args.thresholds) * len(seeds)
    done = 0
    logger.info(f"Plan: {total} runs")

    for ft in args.file_types:
        tr_mask_ft = bundle.ft_train == ft
        te_mask_ft = bundle.ft_test == ft
        ch_mask_ft = bundle.ft_ch == ft
        tb_mask_ft = bundle.ft_test_benign == ft

        # Challenge-mixed slice for this file type: test-benign + challenge-malware.
        X_tb_ft = bundle.X_test_benign[tb_mask_ft]
        y_tb_ft = bundle.y_test_benign[tb_mask_ft]
        X_ch_ft = bundle.X_ch_mal[ch_mask_ft]
        y_ch_ft = bundle.y_ch_mal[ch_mask_ft]
        X_chmix_ft = np.vstack([X_tb_ft, X_ch_ft])
        y_chmix_ft = np.concatenate([y_tb_ft, y_ch_ft])

        X_te_ft = bundle.X_test[te_mask_ft]
        y_te_ft = bundle.y_test[te_mask_ft]

        logger.info(
            f"=== {ft} | train_ft={tr_mask_ft.sum():,}  "
            f"test_ft={te_mask_ft.sum():,}  "
            f"ch_ft={ch_mask_ft.sum():,}  "
            f"chmix_ft={len(y_chmix_ft):,} ==="
        )

        for T in args.thresholds:
            mask_T = filter_by_threshold(bundle.r_train, bundle.y_train, T)
            mask = mask_T & tr_mask_ft
            X_T = bundle.X_train[mask]
            y_T = bundle.y_train[mask]
            logger.info(f"--- {ft} T={T:.3f} n_train={mask.sum():,} "
                        f"mal={(y_T==1).sum():,} ben={(y_T==0).sum():,} ---")

            for seed in seeds:
                fname = f"LightGBM__{ft}__T{T:.3f}__seed{seed}.json"
                result_file = out / "per_run" / fname
                if args.resume and already_done(result_file):
                    done += 1
                    logger.info(f"[SKIP] {fname} ({done}/{total})")
                    continue
                try:
                    t0 = time.time()
                    model = train("LightGBM", X_T, y_T, seed=seed)
                    tt = time.time() - t0
                    p_te = predict_proba("LightGBM", model, X_te_ft)
                    p_ch = predict_proba("LightGBM", model, X_chmix_ft)
                    res = {
                        "classifier": "LightGBM",
                        "file_type": ft,
                        "T": T, "seed": seed,
                        "n_train": int(mask.sum()),
                        "train_time_s": round(tt, 2),
                        "test": compute_metrics(y_te_ft, p_te),
                        "challenge": compute_metrics(y_chmix_ft, p_ch),
                    }
                    write_json_atomic(res, result_file)
                    done += 1
                    logger.info(
                        f"[DONE {done}/{total}] {ft} T={T:.3f} seed={seed} "
                        f"ch_TPR@1%={res['challenge']['TPR@1.0%FPR']*100:.2f}% "
                        f"({tt:.1f}s)"
                    )
                except Exception as e:
                    logger.exception(f"[FAIL] {ft} T={T} seed={seed}: {e}")

    logger.info(f"[DONE] {done}/{total}")


if __name__ == "__main__":
    main()
