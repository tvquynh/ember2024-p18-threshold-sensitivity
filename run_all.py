#!/usr/bin/env python3
"""run_all.py — Master orchestrator for the JISA threshold-sensitivity study.

Runs, in order:
    1. run_distributions       (~2 min, metadata only)
    2. run_cross_classifier    (~16 h, 240 models)
    3. run_lgbm_finegrid       (~9  h, 110 models)
    4. run_weighting           (~8  h, 100 models)
    5. run_per_type            (~4  h, 120 models)
    6. analysis                (seconds)
    7. figures                 (seconds)

Use --prototype for a ~10-min smoke test with small sample sizes.

Cross-platform: calls sub-scripts via `python -m runners.<name>`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PARQUET_DIR_DEFAULT, PATHS, RUN_TAG


STAGES = [
    ("distributions",     ["runners.run_distributions"],   True),
    ("cross_classifier",  ["runners.run_cross_classifier"], True),
    ("lgbm_fine",         ["runners.run_lgbm_finegrid"],   True),
    ("weighting",         ["runners.run_weighting"],       True),
    ("per_type",          ["runners.run_per_type"],        True),
    ("analysis",          ["analysis"],                    False),
    ("figures",           ["figures"],                     False),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", default=PARQUET_DIR_DEFAULT)
    p.add_argument("--output_root", default=PATHS.output_root)
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these stages (names from STAGES)")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Skip these stages")
    p.add_argument("--prototype", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Pass --resume to each training runner")
    return p.parse_args()


def run_stage(name: str, module: str, accepts_parquet: bool, args) -> int:
    cmd = [sys.executable, "-m", module]
    if accepts_parquet:
        cmd += ["--parquet_dir", args.parquet_dir,
                "--output_dir", str(Path(args.output_root) / _stage_output(name))]
        if args.prototype:
            cmd.append("--prototype")
        if args.resume:
            cmd.append("--resume")
    else:
        cmd += ["--output_root", args.output_root]
    print(f"\n========== STAGE: {name} ==========")
    print("  " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    print(f"[{name}] rc={rc} elapsed={dt/60:.1f} min", flush=True)
    return rc


def _stage_output(name: str) -> str:
    return {
        "distributions":    PATHS.distributions,
        "cross_classifier": PATHS.cross_classifier,
        "lgbm_fine":        PATHS.lgbm_fine,
        "weighting":        PATHS.weighting,
        "per_type":         PATHS.per_type,
    }[name]


def main():
    args = parse_args()
    selected = args.only if args.only else [s[0] for s in STAGES]
    print(f"RUN_TAG={RUN_TAG}")
    print(f"parquet_dir={args.parquet_dir}")
    print(f"output_root={args.output_root}")
    print(f"stages: {selected}")

    rc_all = 0
    for name, modules, accepts in STAGES:
        if name not in selected or name in args.skip:
            continue
        for mod in modules:
            rc = run_stage(name, mod, accepts, args)
            if rc != 0:
                print(f"!! stage {name} failed (rc={rc}); continuing to next stage")
                rc_all = rc
    print(f"\n[ALL DONE] overall rc={rc_all}")
    sys.exit(rc_all)


if __name__ == "__main__":
    main()
