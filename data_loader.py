#!/usr/bin/env python3
"""data_loader.py — Load EMBER2024 once, apply threshold filter repeatedly.

Design:
    1. Load full train set (X, y, detection_ratio, file_type) into RAM ONCE.
    2. Load test + challenge once; these never change across thresholds.
    3. `filter_by_threshold(T)` returns a boolean index into the train set.

With 512 GB RAM this is trivially cheap and saves ~45 min of parquet I/O
per full run (110 threshold evaluations × ~25 s per cold load).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl

from ember_v3_schema import TOTAL_DIMS, assert_schema

FEATURE_COLS: List[str] = [f"feature_{i:04d}" for i in range(TOTAL_DIMS)]
META_COLS: List[str] = [
    "sha256", "file_type", "label", "detection_ratio",
    "first_submission_date", "last_analysis_date",
]


@dataclass
class DataBundle:
    """Everything a runner needs, shared across all thresholds."""
    X_train: np.ndarray         # (N, 2568) float32
    y_train: np.ndarray         # (N,) int32
    r_train: np.ndarray         # (N,) float32 detection_ratio
    ft_train: np.ndarray        # (N,) object str, lowercase
    X_test: np.ndarray
    y_test: np.ndarray
    ft_test: np.ndarray
    X_ch_mal: np.ndarray        # challenge malware only
    y_ch_mal: np.ndarray
    ft_ch: np.ndarray
    X_test_benign: np.ndarray   # subset of test with label==0 (for challenge eval)
    y_test_benign: np.ndarray
    ft_test_benign: np.ndarray  # file_type aligned with X_test_benign rows

    @property
    def X_ch_mixed(self) -> np.ndarray:
        return np.vstack([self.X_test_benign, self.X_ch_mal])

    @property
    def y_ch_mixed(self) -> np.ndarray:
        return np.concatenate([self.y_test_benign, self.y_ch_mal])


def _load_parquet(
    parquet_dir: str,
    subset: str,
    file_types: List[str],
    with_features: bool,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    path = Path(parquet_dir) / f"ember2024_{subset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    cols = columns
    if cols is None:
        cols = META_COLS + (FEATURE_COLS if with_features else [])
    df = pl.read_parquet(path, columns=cols)
    if file_types:
        fts = [ft.lower() for ft in file_types]
        df = df.filter(pl.col("file_type").str.to_lowercase().is_in(fts))
    return df


def _extract(df: pl.DataFrame):
    X = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = df["label"].to_numpy().astype(np.int32)
    r = df["detection_ratio"].to_numpy().astype(np.float32)
    ft = np.array([v.lower() for v in df["file_type"].to_list()])
    return X, y, r, ft


def load_all(
    parquet_dir: str,
    file_types: List[str],
    logger: logging.Logger,
    prototype_sizes: Optional[tuple] = None,
) -> DataBundle:
    """Load train / test / challenge. Heavy: ~30 GB peak, ~2 min on SSD."""
    t0 = time.time()
    logger.info(f"[LOAD] parquet_dir={parquet_dir} file_types={file_types}")

    df_tr = _load_parquet(parquet_dir, "train", file_types, with_features=True)
    X_tr, y_tr, r_tr, ft_tr = _extract(df_tr)
    assert_schema(X_tr)
    logger.info(
        f"[LOAD] train: X={X_tr.shape} mal={(y_tr==1).sum():,} "
        f"ben={(y_tr==0).sum():,} r_median(mal)={np.median(r_tr[y_tr==1]):.3f}"
    )
    del df_tr

    df_te = _load_parquet(parquet_dir, "test", file_types, with_features=True)
    X_te, y_te, _, ft_te = _extract(df_te)
    assert_schema(X_te)
    logger.info(f"[LOAD] test: X={X_te.shape}")
    del df_te

    df_ch = _load_parquet(parquet_dir, "challenge", file_types, with_features=True)
    X_ch, y_ch, _, ft_ch = _extract(df_ch)
    assert_schema(X_ch)
    logger.info(f"[LOAD] challenge: X={X_ch.shape} (all malware)")
    del df_ch

    ben_mask = y_te == 0
    X_tb = X_te[ben_mask]
    y_tb = y_te[ben_mask]
    ft_tb = ft_te[ben_mask]
    logger.info(f"[LOAD] test-benign (for challenge mix): {len(y_tb):,}")

    if prototype_sizes is not None:
        X_tr, y_tr, r_tr, ft_tr = _subsample_prototype(
            X_tr, y_tr, r_tr, ft_tr, prototype_sizes[0], logger
        )
        X_te, y_te, ft_te = _subsample_balanced(
            X_te, y_te, ft_te, prototype_sizes[1], logger, tag="test"
        )
        n_tb_keep = min(prototype_sizes[1] // 2, len(X_tb))
        rs_tb = np.random.RandomState(44)
        sel_tb = rs_tb.choice(len(X_tb), n_tb_keep, replace=False)
        X_tb, y_tb, ft_tb = X_tb[sel_tb], y_tb[sel_tb], ft_tb[sel_tb]
        logger.info(f"[LOAD] prototype test-benign: {n_tb_keep:,} "
                    f"(ft mix: "
                    f"win32={(ft_tb=='win32').sum()} "
                    f"win64={(ft_tb=='win64').sum()} "
                    f"dot_net={(ft_tb=='dot_net').sum()})")

    logger.info(f"[LOAD] total elapsed: {time.time()-t0:.1f}s")
    return DataBundle(
        X_train=X_tr, y_train=y_tr, r_train=r_tr, ft_train=ft_tr,
        X_test=X_te, y_test=y_te, ft_test=ft_te,
        X_ch_mal=X_ch, y_ch_mal=y_ch, ft_ch=ft_ch,
        X_test_benign=X_tb, y_test_benign=y_tb, ft_test_benign=ft_tb,
    )


def _subsample_prototype(X, y, r, ft, n, logger):
    rs = np.random.RandomState(42)
    mal = np.where(y == 1)[0]
    ben = np.where(y == 0)[0]
    n_half = n // 2
    sel = np.concatenate([
        rs.choice(mal, min(n_half, len(mal)), replace=False),
        rs.choice(ben, min(n_half, len(ben)), replace=False),
    ])
    rs.shuffle(sel)
    logger.info(f"[LOAD] prototype train: {len(sel):,} ({n_half} mal / {n_half} ben)")
    return X[sel], y[sel], r[sel], ft[sel]


def _subsample_balanced(X, y, ft, n, logger, tag: str):
    rs = np.random.RandomState(43)
    sel = rs.choice(len(y), min(n, len(y)), replace=False)
    logger.info(f"[LOAD] prototype {tag}: {len(sel):,}")
    return X[sel], y[sel], ft[sel]


def filter_by_threshold(
    r_train: np.ndarray, y_train: np.ndarray, T: float
) -> np.ndarray:
    """Return boolean mask: ALL benign + malware with detection_ratio >= T."""
    mal = y_train == 1
    return (~mal) | (mal & (r_train >= T))


def threshold_summary(r_train: np.ndarray, y_train: np.ndarray,
                      thresholds: List[float]) -> List[dict]:
    """Training set composition table after threshold filtering."""
    n_ben = int((y_train == 0).sum())
    n_mal_full = int((y_train == 1).sum())
    out = []
    for T in thresholds:
        mask = filter_by_threshold(r_train, y_train, T)
        n_mal = int(((y_train == 1) & mask).sum())
        out.append({
            "T": T,
            "n_benign": n_ben,
            "n_malware_retained": n_mal,
            "pct_retained": 100.0 * n_mal / n_mal_full if n_mal_full else 0.0,
            "mal_ben_ratio": n_mal / n_ben if n_ben else 0.0,
        })
    return out
