#!/usr/bin/env python3
"""weights.py — Detection-weighted training strategies.

All strategies map (y, r) -> per-sample weight array.
Benign samples always receive weight 1.0 (manuscript Section 3.3).

    uniform:      w = 1.0
    linear:       w = r           (malware only)
    sqrt:         w = sqrt(r)
    log:          w = log(1 + 10r) / log(11)
    binary_high:  w = 1.0 if r >= 0.5 else 0.5
    step_3level:  w = 1.0 if r >= 0.5 else 0.7 if r >= 0.2 else 0.4
"""
from __future__ import annotations

import numpy as np


def weights_for(strategy: str, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown weighting strategy: {strategy}. "
                         f"Valid: {list(_STRATEGIES.keys())}")
    return _STRATEGIES[strategy](y.astype(np.int32), r.astype(np.float64))


def _uniform(y, r):
    return np.ones(len(y), dtype=np.float64)


def _linear(y, r):
    w = np.ones(len(y), dtype=np.float64)
    mal = y == 1
    w[mal] = r[mal]
    return w


def _sqrt(y, r):
    w = np.ones(len(y), dtype=np.float64)
    mal = y == 1
    w[mal] = np.sqrt(np.clip(r[mal], 0.0, 1.0))
    return w


def _log(y, r):
    w = np.ones(len(y), dtype=np.float64)
    mal = y == 1
    w[mal] = np.log1p(10.0 * np.clip(r[mal], 0.0, 1.0)) / np.log(11.0)
    return w


def _binary_high(y, r):
    w = np.ones(len(y), dtype=np.float64)
    mal = y == 1
    low = mal & (r < 0.5)
    w[low] = 0.5
    return w


def _step_3level(y, r):
    w = np.ones(len(y), dtype=np.float64)
    mal = y == 1
    mid = mal & (r < 0.5) & (r >= 0.2)
    low = mal & (r < 0.2)
    w[mid] = 0.7
    w[low] = 0.4
    return w


_STRATEGIES = {
    "uniform":     _uniform,
    "linear":      _linear,
    "sqrt":        _sqrt,
    "log":         _log,
    "binary_high": _binary_high,
    "step_3level": _step_3level,
}


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1, 1, 1])
    r = np.array([0.0, 0.0, 0.05, 0.15, 0.30, 0.70])
    for s in _STRATEGIES:
        print(f"{s:13s} {weights_for(s, y, r).round(3)}")
