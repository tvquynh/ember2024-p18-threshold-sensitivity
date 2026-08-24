#!/usr/bin/env python3
"""Revision-2 controls for the JISA round-2 response.

Three arms, all LightGBM only, all on the same data, seeds and fixed
evaluation populations as the published study, so that every new number is
directly comparable to the existing results.

ARM "canonical"  (Reviewer #1)
    The published study trains LightGBM at learning rate 0.05 without
    bagging, feature subsampling, L2 or class balancing. The EMBER2024
    authors' released lgbm_config.json uses learning rate 0.1 plus all of
    those. This arm repeats the six-point coarse sweep under the authors'
    released configuration, so the sweet-zone claim can be checked against
    the canonical baseline rather than only against ours.
    6 thresholds x 10 seeds = 60 fits.

ARM "matched_size"  (Reviewer #3)
    Raising T does three things at once: it removes low-confidence labels,
    it shrinks the malware pool, and it shifts the class prior. This arm
    isolates the first from the other two. For each target threshold it
    trains on a malware set of exactly the SAME SIZE as that threshold
    retains, but drawn uniformly at random from the baseline-retained pool
    instead of by the r >= T rule. If the random-drop arm reproduces the
    gain, the effect is about training-set size, not label cleanliness.
    3 thresholds x 10 seeds = 30 fits.

ARM "authors_pipeline"  (fidelity check)
    The canonical arm adopts the authors' hyperparameters but fits on the
    whole retained pool, whereas thrember.train_model holds out a stratified
    10% as a monitoring set and therefore fits on 90%. This arm reproduces
    their pipeline end to end -- released configuration AND 90/10 split --
    at T_base only, so the published benchmark can be checked directly.
    1 threshold x 10 seeds = 10 fits.

Usage:
    python runners/run_revision2_controls.py --arm canonical    --output_dir results/rev2_canonical
    python runners/run_revision2_controls.py --arm matched_size --output_dir results/rev2_matched
    (add --prototype for a fast smoke test)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (PE_FILE_TYPES, PARQUET_DIR_DEFAULT, SEEDS, BASELINE_T,
                    PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE, PROTOTYPE_SEEDS,
                    RUN_TAG)
from data_loader import load_all, filter_by_threshold
from classifiers import train, predict_proba
from metrics import (compute_metrics, compute_calibration, tpr_at_fpr,
                     tpr_at_fpr_nearest)
from runners._common import (setup_logger, write_json_atomic, already_done,
                             time_block)

# The threshold grid actually reported in the manuscript (the exploratory
# below-baseline arm was retired before publication, so config.COARSE_THRESHOLDS
# must NOT be used here).
COARSE_ABOVE: list[float] = [0.065, 0.10, 0.15, 0.20, 0.30, 0.50]

# Thresholds for the matched-size control: the sweet-zone interior points the
# paper's recommendation rests on, plus the baseline itself.
#
# T_base is included deliberately. At T = 0.065 the matched-size mask is
# degenerate -- the number to keep equals the size of the baseline-retained
# malware pool, so nothing is dropped and the mask is exactly the baseline mask.
# That gives an in-session parity anchor computed on this machine with this
# library stack, so the control is differenced against a baseline measured here
# rather than against the cluster numbers in the published aggregate.
MATCHED_TARGETS: list[float] = [0.065, 0.10, 0.12, 0.15]

# The EMBER2024 authors' released training configuration
# (examples/lgbm_config.json in the thrember repository). This differs from the
# study's own LGBM_PARAMS, which is the point of this arm.
CANONICAL_LGBM_PARAMS: dict = {
    # Transcribed field for field from examples/lgbm_config.json in the authors'
    # release (github.com/FutureComputing4AI/EMBER2024).
    "objective": "binary",
    "boosting": "gbdt",
    "task": "train",
    "tree_learner": "serial",
    "device_type": "cpu",
    "num_iterations": 500,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "min_sum_hessian_in_leaf": 0.001,
    "learning_rate": 0.1,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "feature_fraction_bynode": 0.9,
    "lambda_l1": 0,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "boost_from_average": True,
    "max_delta_step": 0,
    "sigmoid": 1.0,
    "first_metric_only": True,
    "metric": ["auc", "binary_logloss", "binary_error"],
    "num_threads": 0,
    # Two deliberate deviations from the released file, both disclosed in the
    # manuscript:
    #  * verbosity: the release sets 2; we silence it, which cannot affect fits.
    #  * bagging_seed / feature_fraction_seed: the release pins both to 0, which
    #    would make the bagging and feature draws identical across our ten
    #    seeds and so understate seed-to-seed dispersion. We leave them unset so
    #    LightGBM derives them from `seed`, exactly as the study's own arm does.
    #    This keeps the seeding scheme constant across arms, so the only thing
    #    that differs is the hyperparameters under test.
    "verbosity": -1,
    # feature_pre_filter is absent from the released file, so LightGBM's default
    # (True) applies. Setting it explicitly here documents that and stops
    # classifiers.py from silently substituting the study's own value.
    "feature_pre_filter": True,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", required=True,
                   choices=["canonical", "matched_size", "authors_pipeline"])
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--thresholds", nargs="*", type=float, default=None,
                   help="override the arm's default threshold list")
    p.add_argument("--prototype", action="store_true",
                   help="smoke test: small subsample, 2 seeds")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def evaluate(model, bundle) -> dict:
    """Score the two fixed evaluation populations, exactly as the study does."""
    p_test = predict_proba("LightGBM", model, bundle.X_test)
    p_ch = predict_proba("LightGBM", model, bundle.X_ch_mixed)
    out = {
        "test": compute_metrics(bundle.y_test, p_test),
        "test_calibration": compute_calibration(bundle.y_test, p_test),
        "challenge": compute_metrics(bundle.y_ch_mixed, p_ch),
        "challenge_calibration": compute_calibration(bundle.y_ch_mixed, p_ch),
    }
    # Also record the authors' TPR@FPR convention so a number from their
    # evaluation script and a number from this paper are comparable.
    for split, y, p in (("test", bundle.y_test, p_test),
                        ("challenge", bundle.y_ch_mixed, p_ch)):
        out[split]["TPR@1.0%FPR_nearest"] = tpr_at_fpr_nearest(y, p, 0.01)
        out[split]["TPR@1.0%FPR_atmost"] = tpr_at_fpr(y, p, 0.01)
    return out


def authors_holdout(mask: np.ndarray, y: np.ndarray, seed: int, logger):
    """Reproduce thrember.train_model's stratified 90/10 split.

    model.py:365 calls ``train_test_split(X, y, test_size=0.1, stratify=y)``
    and passes the 10% as ``valid_sets``. With no early-stopping callback the
    validation set cannot affect the fitted model, so the only consequence
    that matters is that the authors fit on 90% of the pool. We reproduce
    that, seeding the split with the run seed (their call leaves it
    unseeded, which would make the arm irreproducible).
    """
    from sklearn.model_selection import train_test_split
    idx = np.where(mask)[0]
    tr, _ = train_test_split(idx, test_size=0.1, stratify=y[idx],
                             random_state=seed)
    out = np.zeros_like(mask)
    out[tr] = True
    logger.info(f"    authors' 90/10 split: fitting on {out.sum():,} of "
                f"{mask.sum():,} ({100 * out.sum() / mask.sum():.1f}%)")
    return out


def matched_size_mask(r_train: np.ndarray, y_train: np.ndarray,
                      n_keep_malware: int, seed: int, T_label: float,
                      logger) -> np.ndarray:
    """Baseline-retained set, with malware randomly thinned to n_keep_malware.

    Benign samples are untouched, exactly as in the real threshold arms. The
    malware kept is a uniform random subset of the malware the BASELINE
    retains -- not of all malware -- so the only difference from the real
    T arm is *which* malware were dropped, never how many.
    """
    base = filter_by_threshold(r_train, y_train, BASELINE_T)
    mal_idx = np.where(base & (y_train == 1))[0]
    if n_keep_malware > mal_idx.size:
        raise ValueError(f"cannot keep {n_keep_malware:,} of {mal_idx.size:,} "
                         f"baseline-retained malware")
    # Independent, reproducible stream per (target threshold, seed).
    rng = np.random.default_rng([int(seed), int(round(T_label * 1000))])
    keep = rng.choice(mal_idx, size=n_keep_malware, replace=False)

    mask = base.copy()
    mask[mal_idx] = False           # drop every baseline-retained malware...
    mask[keep] = True               # ...then put the sampled ones back
    logger.info(f"    matched-size: kept {n_keep_malware:,} of "
                f"{mal_idx.size:,} baseline malware "
                f"(dropped {mal_idx.size - n_keep_malware:,} at random)")
    return mask


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"rev2_{args.arm}", out / "run.log")
    logger.info(f"RUN_TAG={RUN_TAG}  arm={args.arm}")
    logger.info(f"args={vars(args)}")

    seeds = PROTOTYPE_SEEDS if args.prototype else args.seeds
    proto = (PROTOTYPE_TRAIN_SIZE, PROTOTYPE_TEST_SIZE) if args.prototype else None

    with time_block(logger, "data load"):
        bundle = load_all(args.parquet_dir, PE_FILE_TYPES, logger,
                          prototype_sizes=proto)

    if args.arm == "canonical":
        thresholds = args.thresholds or COARSE_ABOVE
        params = CANONICAL_LGBM_PARAMS
        logger.info("Using the EMBER2024 authors' released configuration:")
        for k, v in sorted(params.items()):
            logger.info(f"    {k} = {v}")
    elif args.arm == "authors_pipeline":
        thresholds = args.thresholds or [BASELINE_T]
        params = CANONICAL_LGBM_PARAMS
        logger.info("Reproducing the authors' pipeline end to end: released "
                    "configuration + stratified 90/10 split at T_base only")
        for k, v in sorted(params.items()):
            logger.info(f"    {k} = {v}")
    else:
        thresholds = args.thresholds or MATCHED_TARGETS
        params = None                      # the study's own LGBM_PARAMS
        logger.info("Using the study's own LGBM_PARAMS (params_override=None)")

    # Malware count retained by each real threshold -- the size the matched-size
    # arm must reproduce. Measured from the data, never hard-coded.
    y, r = bundle.y_train, bundle.r_train
    sizes = {T: int((filter_by_threshold(r, y, T) & (y == 1)).sum())
             for T in sorted(set(thresholds) | {BASELINE_T})}
    logger.info("Malware retained per threshold (measured): " +
                ", ".join(f"T={T:.3f}:{n:,}" for T, n in sorted(sizes.items())))

    total = len(thresholds) * len(seeds)
    logger.info(f"Plan: {len(thresholds)} thresholds x {len(seeds)} seeds = {total} fits")

    done = failed = 0
    for T in thresholds:
        logger.info(f"=== {args.arm} | T={T:.3f} ===")
        for seed in seeds:
            tag = f"LightGBM__T{T:.3f}__seed{seed}__{args.arm}"
            result_file = out / "per_run" / f"{tag}.json"
            if args.resume and already_done(result_file):
                done += 1
                logger.info(f"[SKIP] {tag} ({done}/{total})")
                continue
            try:
                if args.arm == "canonical":
                    mask = filter_by_threshold(r, y, T)
                elif args.arm == "authors_pipeline":
                    mask = authors_holdout(filter_by_threshold(r, y, T),
                                           y, seed, logger)
                else:
                    mask = matched_size_mask(r, y, sizes[T], seed, T, logger)

                n_mal = int((mask & (y == 1)).sum())
                n_ben = int((mask & (y == 0)).sum())
                if args.arm == "matched_size" and n_mal != sizes[T]:
                    raise AssertionError(
                        f"matched-size mismatch at T={T}: got {n_mal:,}, "
                        f"expected {sizes[T]:,}")

                t0 = time.time()
                model = train("LightGBM", bundle.X_train[mask], y[mask],
                              seed=seed, params_override=params)
                train_time = time.time() - t0

                res = {
                    "arm": args.arm,
                    "classifier": "LightGBM",
                    "T": T,
                    "seed": seed,
                    "n_train": int(mask.sum()),
                    "n_train_malware": n_mal,
                    "n_train_benign": n_ben,
                    "train_time_s": round(train_time, 2),
                    "params": params if params is not None else "study LGBM_PARAMS",
                    **evaluate(model, bundle),
                }
                write_json_atomic(res, result_file)
                done += 1
                logger.info(
                    f"[DONE {done}/{total}] T={T:.3f} seed={seed} "
                    f"mal={n_mal:,} ch_TPR@1%="
                    f"{res['challenge'].get('TPR@1.0%FPR', float('nan'))*100:.2f}% "
                    f"({res['train_time_s']}s)")
            except Exception as e:                       # noqa: BLE001
                failed += 1
                logger.exception(f"[FAIL] T={T} seed={seed}: {e}")

    logger.info(f"[COMPLETE] {done}/{total} succeeded, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
