#!/usr/bin/env python3
"""run_seed.py — Single-seed worker for P18 cluster execution.

Runs all 4 stages (cross_classifier, lgbm_fine, weighting, per_type)
plus the distributions snapshot for ONE seed value. Each compute node
in the SLURM array calls this with a different --seed.

Output structure (per seed):
    <output_dir>/
        log.txt                                          per-seed log
        done.json                                        marker after all stages
        distributions/snapshot.json                      computed once (idempotent)
        cross_classifier/per_run/<clf>__T<T>.json        36 files
        lgbm_fine/per_run/T<T>.json                      14 files
        weighting/per_run/<clf>__<strategy>.json         10 files
        per_type/per_run/<ft>__T<T>.json                 12 files

Usage:
    # On a typical compute node (~60 cores):
    python run_seed.py --seed 123 \\
                       --output_dir <shared_results>/seed_123 \\
                       --parquet_dir <ember2024_parquet_dir> \\
                       --num_threads 56

    # On a higher-core head node (~128 cores):
    python run_seed.py --seed 42 \\
                       --output_dir <shared_results>/seed_42 \\
                       --parquet_dir <ember2024_parquet_dir> \\
                       --num_threads 120

    # Smoke test (~10 min on a single node, this seed only):
    python run_seed.py --seed 42 --output_dir /tmp/p18_smoke --prototype \\
                       --parquet_dir <ember2024_parquet_dir> --num_threads 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    BASELINE_T, COARSE_THRESHOLDS, FINE_THRESHOLDS, PER_TYPE_THRESHOLDS,
    CLASSIFIER_NAMES, MLP_PARAMS, PE_FILE_TYPES, PARQUET_DIR_DEFAULT,
    PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE,
    WEIGHTING_STRATEGIES, WEIGHTING_STRATEGIES_XGB, RUN_TAG,
)
from data_loader import load_all, filter_by_threshold, threshold_summary
from classifiers import train, predict_proba, stratified_val_split
from weights import weights_for
from metrics import compute_metrics, compute_calibration
from runners._common import (setup_logger, write_json_atomic, read_json,
                             already_done, time_block)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True,
                   help="Single seed value (e.g. 42, 123, ...)")
    p.add_argument("--output_dir", required=True,
                   help="Per-seed output dir, e.g. /srv/nfs/results/.../seed_42")
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--num_threads", type=int, default=None,
                   help="Override n_jobs for LGB/XGB/RF and torch threads")
    p.add_argument("--prototype", action="store_true",
                   help="Smoke test with ~20K train, 1 seed (--seed)")
    p.add_argument("--resume", action="store_true",
                   help="Skip per-run files that already exist")
    p.add_argument("--stages", nargs="*",
                   default=["distributions", "cross_classifier",
                            "lgbm_fine", "weighting", "per_type"],
                   help="Subset of stages to run")
    return p.parse_args()


def configure_threads(num_threads: int | None, logger):
    """Override LGB/XGB n_jobs and torch threads via env + monkey-patching params."""
    if num_threads is None or num_threads <= 0:
        return
    # LightGBM / XGBoost / sklearn read via env vars + n_jobs param
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    # Patch config defaults so trainers pick up our value
    from config import LGBM_PARAMS, XGB_PARAMS, XGB_PARAMS_DETERMINISTIC, RF_PARAMS
    LGBM_PARAMS["n_jobs"] = num_threads
    XGB_PARAMS["n_jobs"] = num_threads
    XGB_PARAMS_DETERMINISTIC["n_jobs"] = num_threads
    RF_PARAMS["n_jobs"] = num_threads
    # Torch (lazy import — only when MLP runs)
    try:
        import torch
        torch.set_num_threads(num_threads)
    except ImportError:
        pass
    logger.info(f"Configured n_jobs={num_threads} for LGB/XGB/RF/torch")


# ── Stage 1: distributions ───────────────────────────────────────────────────
def stage_distributions(bundle, output_dir: Path, parquet_dir: str, logger):
    """Detection-ratio distribution snapshot. Seed-independent.

    DataBundle does NOT carry challenge detection_ratio (challenge samples
    are all malware by construction; ratio is unused at training time).
    So we re-load the meta columns from parquet here. Cheap (~5 s).
    """
    out = output_dir / "distributions"
    out.mkdir(parents=True, exist_ok=True)
    snap_file = out / "snapshot.json"
    if snap_file.exists():
        logger.info(f"[distributions] snapshot exists, skip")
        return

    import polars as pl

    def _percentiles(r):
        r = np.asarray(r, dtype=np.float64)
        if len(r) == 0:
            return {"n": 0}
        return {
            "n": int(len(r)),
            "mean": float(r.mean()),
            "median": float(np.median(r)),
            "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            "p25": float(np.percentile(r, 25)),
            "p75": float(np.percentile(r, 75)),
            "p90": float(np.percentile(r, 90)),
            "below_0.10": float((r < 0.10).mean()),
            "below_0.15": float((r < 0.15).mean()),
            "below_0.20": float((r < 0.20).mean()),
        }

    # Load meta-only parquet for distributions (challenge needs r_ch).
    pe_lower = [f.lower() for f in PE_FILE_TYPES]
    df_tr = pl.read_parquet(
        Path(parquet_dir) / "ember2024_train.parquet",
        columns=["file_type", "label", "detection_ratio"],
    ).filter(pl.col("file_type").str.to_lowercase().is_in(pe_lower))
    df_ch = pl.read_parquet(
        Path(parquet_dir) / "ember2024_challenge.parquet",
        columns=["file_type", "label", "detection_ratio"],
    ).filter(pl.col("file_type").str.to_lowercase().is_in(pe_lower))

    r_tr = df_tr["detection_ratio"].to_numpy().astype(np.float32)
    y_tr = df_tr["label"].to_numpy().astype(np.int32)
    ft_tr = np.array([v.lower() for v in df_tr["file_type"].to_list()])
    r_ch = df_ch["detection_ratio"].to_numpy().astype(np.float32)
    ft_ch = np.array([v.lower() for v in df_ch["file_type"].to_list()])

    snap = {
        "run_tag": RUN_TAG,
        "global": {
            "train_malware":     _percentiles(r_tr[y_tr == 1]),
            "challenge_malware": _percentiles(r_ch),
        },
        "per_file_type": {
            ft: {
                "train_malware":     _percentiles(r_tr[(y_tr == 1) & (ft_tr == ft)]),
                "challenge_malware": _percentiles(r_ch[ft_ch == ft]),
            } for ft in PE_FILE_TYPES
        },
        "threshold_data_reduction": threshold_summary(
            r_tr, y_tr, sorted(set(COARSE_THRESHOLDS + FINE_THRESHOLDS))
        ),
        "histogram": {
            "train_malware_ratios": np.histogram(
                r_tr[y_tr == 1], bins=np.linspace(0, 1, 51)
            )[0].tolist(),
            "challenge_malware_ratios": np.histogram(
                r_ch, bins=np.linspace(0, 1, 51)
            )[0].tolist(),
            "bin_edges": np.linspace(0, 1, 51).tolist(),
        },
    }
    write_json_atomic(snap, snap_file)
    tr = snap["global"]["train_malware"]
    ch = snap["global"]["challenge_malware"]
    logger.info(f"[distributions] snapshot at {snap_file}")
    logger.info(
        f"  train mal median={tr['median']:.3f}  "
        f"challenge mal median={ch['median']:.3f}  "
        f"ratio={tr['median']/max(ch['median'], 1e-6):.1f}x"
    )


# ── Stage 2: cross_classifier ────────────────────────────────────────────────
def stage_cross_classifier(bundle, seed: int, output_dir: Path,
                            resume: bool, logger):
    out = output_dir / "cross_classifier"
    (out / "per_run").mkdir(parents=True, exist_ok=True)
    log_lines = []

    plan = [(clf, T) for T in COARSE_THRESHOLDS for clf in CLASSIFIER_NAMES]
    total = len(plan)
    logger.info(f"[cross_classifier] Plan: {total} models "
                f"({len(CLASSIFIER_NAMES)} clf × {len(COARSE_THRESHOLDS)} T × 1 seed)")

    done = 0
    for T in COARSE_THRESHOLDS:
        mask = filter_by_threshold(bundle.r_train, bundle.y_train, T)
        n_mal = int(((bundle.y_train == 1) & mask).sum())
        n_ben = int((bundle.y_train == 0).sum())
        logger.info(f"  T={T:.3f} | train mal={n_mal:,} ben={n_ben:,}")
        X_T = bundle.X_train[mask]
        y_T = bundle.y_train[mask]

        for clf_name in CLASSIFIER_NAMES:
            fname = f"{clf_name}__T{T:.3f}.json"
            result_file = out / "per_run" / fname
            if resume and already_done(result_file):
                done += 1
                logger.info(f"  [SKIP] {fname} ({done}/{total})")
                continue
            try:
                X_tr_use, y_tr_use, X_val, y_val = X_T, y_T, None, None
                if clf_name == "MLP":
                    X_tr_use, y_tr_use, X_val, y_val = stratified_val_split(
                        X_T, y_T, MLP_PARAMS["val_fraction"], seed
                    )
                t0 = time.time()
                model = train(clf_name, X_tr_use, y_tr_use, seed=seed,
                              X_val=X_val, y_val=y_val)
                tt = time.time() - t0
                p_te = predict_proba(clf_name, model, bundle.X_test)
                p_ch = predict_proba(clf_name, model, bundle.X_ch_mixed)
                res = {
                    "classifier": clf_name,
                    "T": T, "seed": seed,
                    "n_train": int(len(y_tr_use)),
                    "n_val": int(len(y_val)) if y_val is not None else 0,
                    "train_time_s": round(tt, 2),
                    "test": compute_metrics(bundle.y_test, p_te),
                    "test_calibration": compute_calibration(bundle.y_test, p_te),
                    "challenge": compute_metrics(bundle.y_ch_mixed, p_ch),
                    "challenge_calibration": compute_calibration(bundle.y_ch_mixed, p_ch),
                }
                write_json_atomic(res, result_file)
                done += 1
                logger.info(
                    f"  [DONE {done}/{total}] {clf_name} T={T:.3f} "
                    f"ch_TPR@1%={res['challenge']['TPR@1.0%FPR']*100:.2f}% "
                    f"({tt:.1f}s)"
                )
            except Exception as e:
                logger.exception(f"  [FAIL] {clf_name} T={T:.3f}: {e}")

    write_json_atomic({"done": done, "total": total, "seed": seed}, out / "done.json")
    logger.info(f"[cross_classifier] {done}/{total} done")


# ── Stage 3: lgbm_fine ───────────────────────────────────────────────────────
def stage_lgbm_fine(bundle, seed: int, output_dir: Path,
                     resume: bool, logger):
    out = output_dir / "lgbm_fine"
    (out / "per_run").mkdir(parents=True, exist_ok=True)
    total = len(FINE_THRESHOLDS)
    logger.info(f"[lgbm_fine] Plan: {total} models (LightGBM × {len(FINE_THRESHOLDS)} T × 1 seed)")

    done = 0
    for T in FINE_THRESHOLDS:
        fname = f"T{T:.3f}.json"
        result_file = out / "per_run" / fname
        if resume and already_done(result_file):
            done += 1
            logger.info(f"  [SKIP] {fname} ({done}/{total})")
            continue
        mask = filter_by_threshold(bundle.r_train, bundle.y_train, T)
        X_T, y_T = bundle.X_train[mask], bundle.y_train[mask]
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
                f"  [DONE {done}/{total}] T={T:.3f} "
                f"ch_TPR@1%={res['challenge']['TPR@1.0%FPR']*100:.2f}% "
                f"({tt:.1f}s)"
            )
        except Exception as e:
            logger.exception(f"  [FAIL] T={T:.3f}: {e}")

    write_json_atomic({"done": done, "total": total, "seed": seed}, out / "done.json")
    logger.info(f"[lgbm_fine] {done}/{total} done")


# ── Stage 4: weighting ───────────────────────────────────────────────────────
def stage_weighting(bundle, seed: int, output_dir: Path,
                     resume: bool, logger):
    out = output_dir / "weighting"
    (out / "per_run").mkdir(parents=True, exist_ok=True)

    plan = (
        [("LightGBM", s) for s in WEIGHTING_STRATEGIES]
        + [("XGBoost",  s) for s in WEIGHTING_STRATEGIES_XGB]
    )
    total = len(plan)
    logger.info(f"[weighting] Plan: {total} models @ T={BASELINE_T} (1 seed)")

    # Training set = baseline T (matches "uniform" row from cross_classifier).
    mask = filter_by_threshold(bundle.r_train, bundle.y_train, BASELINE_T)
    X = bundle.X_train[mask]
    y = bundle.y_train[mask]
    r = bundle.r_train[mask]

    done = 0
    for clf_name, strategy in plan:
        fname = f"{clf_name}__{strategy}.json"
        result_file = out / "per_run" / fname
        if resume and already_done(result_file):
            done += 1
            logger.info(f"  [SKIP] {fname} ({done}/{total})")
            continue
        try:
            w = weights_for(strategy, y, r)
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
                f"  [DONE {done}/{total}] {clf_name} {strategy} "
                f"ch_TPR@1%={res['challenge']['TPR@1.0%FPR']*100:.2f}% "
                f"({tt:.1f}s)"
            )
        except Exception as e:
            logger.exception(f"  [FAIL] {clf_name} {strategy}: {e}")

    write_json_atomic({"done": done, "total": total, "seed": seed}, out / "done.json")
    logger.info(f"[weighting] {done}/{total} done")


# ── Stage 5: per_type ────────────────────────────────────────────────────────
def stage_per_type(bundle, seed: int, output_dir: Path,
                    resume: bool, logger):
    out = output_dir / "per_type"
    (out / "per_run").mkdir(parents=True, exist_ok=True)

    total = len(PE_FILE_TYPES) * len(PER_TYPE_THRESHOLDS)
    logger.info(f"[per_type] Plan: {total} models "
                f"({len(PE_FILE_TYPES)} ft × {len(PER_TYPE_THRESHOLDS)} T × 1 seed)")

    done = 0
    for ft in PE_FILE_TYPES:
        tr_mask_ft = bundle.ft_train == ft
        te_mask_ft = bundle.ft_test == ft
        ch_mask_ft = bundle.ft_ch == ft
        tb_mask_ft = bundle.ft_test_benign == ft

        X_tb_ft = bundle.X_test_benign[tb_mask_ft]
        y_tb_ft = bundle.y_test_benign[tb_mask_ft]
        X_ch_ft = bundle.X_ch_mal[ch_mask_ft]
        y_ch_ft = bundle.y_ch_mal[ch_mask_ft]
        X_chmix_ft = np.vstack([X_tb_ft, X_ch_ft])
        y_chmix_ft = np.concatenate([y_tb_ft, y_ch_ft])

        X_te_ft = bundle.X_test[te_mask_ft]
        y_te_ft = bundle.y_test[te_mask_ft]

        for T in PER_TYPE_THRESHOLDS:
            fname = f"{ft}__T{T:.3f}.json"
            result_file = out / "per_run" / fname
            if resume and already_done(result_file):
                done += 1
                logger.info(f"  [SKIP] {fname} ({done}/{total})")
                continue
            mask_T = filter_by_threshold(bundle.r_train, bundle.y_train, T)
            mask = mask_T & tr_mask_ft
            X_T = bundle.X_train[mask]
            y_T = bundle.y_train[mask]
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
                    f"  [DONE {done}/{total}] {ft} T={T:.3f} "
                    f"ch_TPR@1%={res['challenge']['TPR@1.0%FPR']*100:.2f}% "
                    f"({tt:.1f}s)"
                )
            except Exception as e:
                logger.exception(f"  [FAIL] {ft} T={T:.3f}: {e}")

    write_json_atomic({"done": done, "total": total, "seed": seed}, out / "done.json")
    logger.info(f"[per_type] {done}/{total} done")


# ── Main ─────────────────────────────────────────────────────────────────────
STAGE_FNS = {
    "distributions":    stage_distributions,
    "cross_classifier": stage_cross_classifier,
    "lgbm_fine":        stage_lgbm_fine,
    "weighting":        stage_weighting,
    "per_type":         stage_per_type,
}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"seed_{args.seed}", output_dir / "log.txt")

    logger.info("=" * 60)
    logger.info(f"P18 run_seed.py | RUN_TAG={RUN_TAG}")
    logger.info(f"  seed={args.seed}  prototype={args.prototype}  resume={args.resume}")
    logger.info(f"  output_dir={output_dir}")
    logger.info(f"  parquet_dir={args.parquet_dir}")
    logger.info(f"  stages={args.stages}")
    logger.info("=" * 60)

    configure_threads(args.num_threads, logger)

    proto_sizes = (PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE) if args.prototype else None

    t_overall = time.time()
    with time_block(logger, "data load"):
        bundle = load_all(args.parquet_dir, PE_FILE_TYPES, logger,
                          prototype_sizes=proto_sizes)

    stage_durations = {}
    for stage_name in args.stages:
        if stage_name not in STAGE_FNS:
            logger.warning(f"Unknown stage '{stage_name}' — skipped")
            continue
        t_stage = time.time()
        try:
            fn = STAGE_FNS[stage_name]
            if stage_name == "distributions":
                fn(bundle, output_dir, args.parquet_dir, logger)
            else:
                fn(bundle, args.seed, output_dir, args.resume, logger)
            stage_durations[stage_name] = round((time.time() - t_stage) / 60, 2)
        except Exception as e:
            logger.exception(f"[STAGE FAIL] {stage_name}: {e}")
            stage_durations[stage_name] = -1

    overall_min = round((time.time() - t_overall) / 60, 2)
    summary = {
        "seed": args.seed,
        "prototype": args.prototype,
        "run_tag": RUN_TAG,
        "stages_requested": args.stages,
        "stage_durations_min": stage_durations,
        "overall_min": overall_min,
        "num_threads": args.num_threads,
    }
    write_json_atomic(summary, output_dir / "done.json")
    logger.info("=" * 60)
    logger.info(f"[ALL STAGES DONE] seed={args.seed} | "
                f"overall={overall_min} min | "
                f"per stage min: {stage_durations}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
