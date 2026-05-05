#!/usr/bin/env python3
"""Distribution snapshot — training vs challenge detection-ratio stats.

No training. Produces the numbers for manuscript Figure 1 and the
distributional-mismatch narrative (section 5.1).

Output: results/distributions/snapshot.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (PARQUET_DIR_DEFAULT, PE_FILE_TYPES, PATHS, COARSE_THRESHOLDS,
                    RUN_TAG)
from data_loader import load_all, threshold_summary, FEATURE_COLS
from runners._common import setup_logger, write_json_atomic


def describe(r: np.ndarray) -> dict:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--output_dir", default=f"{PATHS.output_root}/{PATHS.distributions}")
    p.add_argument("--prototype", action="store_true",
                   help="No-op for this stage (distributions runs on metadata only).")
    p.add_argument("--resume", action="store_true",
                   help="No-op for this stage (output is a single snapshot.json).")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("distributions", out / "run.log")
    logger.info(f"RUN_TAG={RUN_TAG}")

    # Need detection_ratio from parquet — we load_all to stay consistent but
    # this is wasteful; acceptable since it runs exactly once per project.
    # Alternative: load only meta columns via polars if we later care.
    import polars as pl
    from data_loader import _load_parquet, _extract

    # Minimal load (meta-only) for distributions. We still need features if we
    # pass through `load_all`, so use the lightweight path:
    df_tr = pl.read_parquet(
        Path(args.parquet_dir) / "ember2024_train.parquet",
        columns=["file_type", "label", "detection_ratio"],
    ).filter(pl.col("file_type").str.to_lowercase().is_in([f.lower() for f in PE_FILE_TYPES]))
    df_ch = pl.read_parquet(
        Path(args.parquet_dir) / "ember2024_challenge.parquet",
        columns=["file_type", "label", "detection_ratio"],
    ).filter(pl.col("file_type").str.to_lowercase().is_in([f.lower() for f in PE_FILE_TYPES]))

    r_tr = df_tr["detection_ratio"].to_numpy().astype(np.float32)
    y_tr = df_tr["label"].to_numpy().astype(np.int32)
    ft_tr = np.array([v.lower() for v in df_tr["file_type"].to_list()])
    r_ch = df_ch["detection_ratio"].to_numpy().astype(np.float32)
    ft_ch = np.array([v.lower() for v in df_ch["file_type"].to_list()])

    snapshot = {
        "run_tag": RUN_TAG,
        "global": {
            "train_malware": describe(r_tr[y_tr == 1]),
            "challenge_malware": describe(r_ch),  # challenge is all malware
        },
        "per_file_type": {},
        "threshold_data_reduction": threshold_summary(
            r_tr, y_tr, COARSE_THRESHOLDS
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

    for ft in PE_FILE_TYPES:
        snapshot["per_file_type"][ft] = {
            "train_malware": describe(r_tr[(y_tr == 1) & (ft_tr == ft)]),
            "challenge_malware": describe(r_ch[ft_ch == ft]),
        }

    write_json_atomic(snapshot, out / "snapshot.json")
    logger.info(f"[DONE] distributions snapshot at {out / 'snapshot.json'}")
    # Compact summary for console
    tr = snapshot["global"]["train_malware"]
    ch = snapshot["global"]["challenge_malware"]
    logger.info(f"Train mal: median={tr['median']:.3f}  Challenge mal: median={ch['median']:.3f}"
                f"  ratio={tr['median']/max(ch['median'],1e-6):.1f}x")


if __name__ == "__main__":
    main()
