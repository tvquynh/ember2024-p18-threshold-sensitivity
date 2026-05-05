#!/usr/bin/env python3
"""_common.py — Shared utilities for runners.

Logging, JSON IO, atomic writes, path helpers.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def write_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Unserializable: {type(o)}")


def run_name(classifier: str, T: float, seed: int,
             suffix: str = "") -> str:
    s = f"{classifier}__T{T:.3f}__seed{seed}"
    if suffix:
        s += f"__{suffix}"
    return s


def already_done(result_file: Path) -> bool:
    return result_file.exists() and result_file.stat().st_size > 10


def time_block(logger, label: str):
    class _T:
        def __enter__(self_inner):
            self_inner.t0 = time.time()
            logger.info(f"[TIME] {label} started")
            return self_inner
        def __exit__(self_inner, *exc):
            dt = time.time() - self_inner.t0
            logger.info(f"[TIME] {label} finished in {dt:.1f}s")
    return _T()
