#!/usr/bin/env python3
"""config.py — Frozen hyperparameters for the threshold-sensitivity study.

All numbers here match the JISA manuscript. Do NOT mutate at runtime;
override via CLI flags if needed. Any change requires a bump in RUN_TAG
so prior results remain traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

RUN_TAG = "jisa_v2_2026_04"

# ── Data ─────────────────────────────────────────────────────────────────────
PARQUET_DIR_DEFAULT = "E:/project_data/parquet_clean-week"
PE_FILE_TYPES: List[str] = ["win32", "win64", "dot_net"]

# ── Seeds (10 fixed, published in manuscript) ────────────────────────────────
SEEDS: List[int] = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
PROTOTYPE_SEEDS: List[int] = [42, 123]

# ── Threshold grids ──────────────────────────────────────────────────────────
# Baseline (EMBER2024 uses k=5 / ~2.68M scanner span → ratio ~0.065).
# This matches Joyce et al. KDD'25 author training pattern.
BASELINE_T: float = 0.065

# Below-baseline grid: RETIRED. An early draft swept T below the EMBER2024
# default (T = 0.0 / 0.02 / 0.04) as a "does filtering itself hurt?" control.
# It was dropped from the study because only ~1.2% of EMBER2024 training
# malware falls below T_base = 0.065, so those thresholds change the training
# set by <1.5% and cannot produce informative contrasts. Kept here, empty, so
# that the retirement is explicit rather than silent.
BELOW_BASELINE_THRESHOLDS: List[float] = []

# Coarse grid (all classifiers): 6 thresholds at and above the baseline.
# 4 classifiers × 6 thresholds × 10 seeds = 240 training runs.
COARSE_ABOVE_THRESHOLDS: List[float] = [0.065, 0.10, 0.15, 0.20, 0.30, 0.50]
COARSE_THRESHOLDS: List[float] = sorted(BELOW_BASELINE_THRESHOLDS + COARSE_ABOVE_THRESHOLDS)

# Fine grid (LightGBM only): adds intermediate thresholds for variance analysis.
# 6 coarse + 5 interior = 11 thresholds × 10 seeds = 110 training runs.
FINE_EXTRA_THRESHOLDS: List[float] = [0.08, 0.12, 0.18, 0.25, 0.40]
FINE_THRESHOLDS: List[float] = sorted(set(COARSE_THRESHOLDS + FINE_EXTRA_THRESHOLDS))

# Sweet zone from the manuscript (used for the shaded region in figures).
# All five thresholds in [0.08, 0.18] are Holm-significant improvements over
# the baseline on the LightGBM fine grid; T = 0.20 is the non-significant
# boundary point.
SWEET_ZONE: Tuple[float, float] = (0.08, 0.18)

# ── Classifier parameters ────────────────────────────────────────────────────
# LightGBM: 500 rounds, 64 leaves, 100 min.data/leaf, lr 0.05
LGBM_PARAMS: Dict = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "learning_rate": 0.05,
    "feature_pre_filter": False,
    "verbose": -1,
    "n_jobs": -1,
}

# XGBoost: 500 rounds, max_depth 8, min_child_weight 100, lr 0.05, hist method.
# Added subsample+colsample to restore seed-dependent variance (manuscript W2).
XGB_PARAMS: Dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 500,
    "max_depth": 8,
    "min_child_weight": 100,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0,
    "n_jobs": -1,
}

# XGBoost deterministic mode (for reproducing v1 results if needed).
XGB_PARAMS_DETERMINISTIC: Dict = {
    **{k: v for k, v in XGB_PARAMS.items() if k not in ("subsample", "colsample_bytree")},
}

# Random Forest: 500 trees, max_depth 20, 100 min.samples/leaf.
RF_PARAMS: Dict = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_leaf": 100,
    "n_jobs": -1,
    "bootstrap": True,
}

# MLP: 512-256-128, dropout 0.3, lr 0.001, batch 4096, 20 epochs, ES patience 5.
MLP_PARAMS: Dict = {
    "hidden_sizes": [512, 256, 128],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 4096,
    "epochs": 20,
    "early_stopping_patience": 5,
    "val_fraction": 0.10,
}

CLASSIFIER_NAMES: List[str] = ["LightGBM", "XGBoost", "RandomForest", "MLP"]

# ── Detection-weighted training strategies ───────────────────────────────────
# All strategies apply ONLY to malware samples; benign receive uniform weight 1.
WEIGHTING_STRATEGIES: List[str] = [
    "uniform", "linear", "sqrt", "log", "binary_high", "step_3level",
]

# Cross-classifier subset (manuscript section 5.4).
WEIGHTING_STRATEGIES_XGB: List[str] = ["uniform", "linear", "sqrt", "log"]

# ── Per-file-type analysis ───────────────────────────────────────────────────
# Thresholds evaluated per file type (manuscript Table 5).
PER_TYPE_THRESHOLDS: List[float] = [0.065, 0.10, 0.15, 0.20]

# ── Evaluation ───────────────────────────────────────────────────────────────
# Multiple operating points for "What FPR can production tolerate?" analysis
# (added 2026-04-25 per Q1 reviewer-anticipated detail).
TARGET_FPRS: List[float] = [0.001, 0.01, 0.05]
PRIMARY_METRIC: str = "TPR@1.0%FPR"
PRIMARY_SPLIT: str = "challenge"

# Calibration analysis: number of bins for reliability diagram + ECE.
CALIBRATION_BINS: int = 10

# ── Statistical test ─────────────────────────────────────────────────────────
ALPHA: float = 0.05

# ── Figures style ────────────────────────────────────────────────────────────
FIG_DPI: int = 300
COLOR_LGBM: str = "#D62728"
COLOR_XGB: str = "#FF7F0E"
COLOR_RF: str = "#2CA02C"
COLOR_MLP: str = "#1F77B4"
COLOR_BASELINE: str = "#7F7F7F"
COLOR_SWEET: str = "#F7D794"

CLASSIFIER_COLORS: Dict[str, str] = {
    "LightGBM":      COLOR_LGBM,
    "XGBoost":       COLOR_XGB,
    "RandomForest":  COLOR_RF,
    "MLP":           COLOR_MLP,
}

# ── Prototype (smoke-test) budget ────────────────────────────────────────────
PROTOTYPE_TRAIN_SIZE: int = 20_000
PROTOTYPE_TEST_SIZE: int = 5_000

# ── Output layout ────────────────────────────────────────────────────────────
# Each runner writes to results/<experiment>/... as per-run JSON plus a
# summary.json aggregator. Crash-safe: per-seed files written atomically.
@dataclass
class Paths:
    output_root: str = "results"
    cross_classifier: str = "cross_classifier"
    lgbm_fine: str = "lgbm_fine"
    weighting: str = "weighting"
    per_type: str = "per_type"
    figures: str = "figures"
    distributions: str = "distributions"
    tables: str = "tables"

PATHS = Paths()
