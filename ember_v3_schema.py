#!/usr/bin/env python3
"""EMBER2024 v3 feature schema — canonical source of truth.

Vendored from the project-wide module so this paper folder is self-contained.
Any change to the schema must happen upstream first.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import numpy as np

FEATURE_GROUPS: "OrderedDict[str, Dict]" = OrderedDict([
    ("GFI",  {"name": "General File Info",       "start": 0,    "end": 7,    "dim": 7}),
    ("BH",   {"name": "Byte Histogram",          "start": 7,    "end": 263,  "dim": 256}),
    ("BEH",  {"name": "Byte-Entropy Histogram",  "start": 263,  "end": 519,  "dim": 256}),
    ("STR",  {"name": "Strings",                 "start": 519,  "end": 696,  "dim": 177}),
    ("HDR",  {"name": "PE Header",               "start": 696,  "end": 770,  "dim": 74}),
    ("SEC",  {"name": "Section Information",     "start": 770,  "end": 994,  "dim": 224}),
    ("IMP",  {"name": "Imports",                 "start": 994,  "end": 2276, "dim": 1282}),
    ("EXP",  {"name": "Exports",                 "start": 2276, "end": 2405, "dim": 129}),
    ("DD",   {"name": "Data Directories",        "start": 2405, "end": 2439, "dim": 34}),
    ("RH",   {"name": "Rich Header",             "start": 2439, "end": 2472, "dim": 33}),
    ("AUTH", {"name": "Authenticode Signature",  "start": 2472, "end": 2480, "dim": 8}),
    ("WARN", {"name": "PE Format Warnings",      "start": 2480, "end": 2568, "dim": 88}),
])

TOTAL_DIMS: int = 2568
SIZE_FEATURE_IDX: int = 0

# Author reference (Joyce et al. KDD'25, thrember/model.py).
CATEGORICAL_FEATURES = [2, 3, 4, 5, 6, 701, 702]


def group_slice(code: str) -> slice:
    g = FEATURE_GROUPS[code]
    return slice(g["start"], g["end"])


def assert_schema(X: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    if X.shape[1] != TOTAL_DIMS:
        raise ValueError(
            f"Feature dim mismatch: expected {TOTAL_DIMS}, got {X.shape[1]}."
        )


if __name__ == "__main__":
    X = np.zeros((2, TOTAL_DIMS), dtype=np.float32)
    assert_schema(X)
    assert X[:, group_slice("BEH")].shape == (2, 256)
    print("[OK] Schema self-test passed.")
