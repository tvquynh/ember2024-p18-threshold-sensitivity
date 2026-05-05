#!/usr/bin/env python3
"""aggregate.py — Merge per-seed results from cluster array into final outputs.

Reads:
    <seeds_dir>/seed_<N>/<stage>/per_run/*.json     (10 seeds × 4 stages)
    <seeds_dir>/seed_<N>/distributions/snapshot.json (any seed; first wins)

Writes:
    <output_dir>/distributions/snapshot.json
    <output_dir>/<stage>/summary.json     (mean/std/CI per (clf, T) across seeds)
    <output_dir>/<stage>/stats.json       (Friedman + Wilcoxon vs baseline)
    <output_dir>/tables/*.csv
    <output_dir>/figures/fig{1..4}.{pdf,png}

Run on the cluster head node AFTER all 10 seed jobs report done.json:

    python aggregate.py --seeds_dir <shared_results_dir> \\
                        --output_dir <shared_results_dir>/aggregated
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PATHS, RUN_TAG
from runners._common import setup_logger, read_json, write_json_atomic
import analysis
import figures


SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def discover_seeds(seeds_dir: Path) -> list[tuple[int, Path]]:
    """Return [(seed_value, seed_dir_path)] for seeds whose done.json exists."""
    out = []
    for child in sorted(seeds_dir.iterdir()):
        if not child.is_dir():
            continue
        m = SEED_DIR_RE.match(child.name)
        if not m:
            continue
        seed = int(m.group(1))
        if (child / "done.json").exists():
            out.append((seed, child))
    return out


def collect_distributions(seed_dirs: list[tuple[int, Path]],
                          dest_dir: Path, logger) -> bool:
    """Copy snapshot.json from first seed that has one. Distributions are
    seed-independent, so any one is fine. Validate that all match."""
    snapshots = []
    for seed, d in seed_dirs:
        snap = d / "distributions" / "snapshot.json"
        if snap.exists():
            snapshots.append((seed, snap))
    if not snapshots:
        logger.warning("No distributions/snapshot.json found in any seed dir")
        return False
    seed0, src = snapshots[0]
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest_dir / "snapshot.json")
    logger.info(f"  distributions snapshot copied from seed_{seed0}")
    # Sanity check: train_malware.median should match across all seeds
    if len(snapshots) > 1:
        ref = read_json(src)["global"]["train_malware"]["median"]
        for seed, p in snapshots[1:]:
            v = read_json(p)["global"]["train_malware"]["median"]
            if abs(v - ref) > 1e-6:
                logger.warning(
                    f"  mismatch in distributions: seed_{seed0} median={ref} "
                    f"vs seed_{seed} median={v}"
                )
    return True


def materialize_stage_per_run(seed_dirs: list[tuple[int, Path]],
                              stage_subdir: str,
                              merge_dir: Path,
                              logger) -> int:
    """Copy seed_*/<stage>/per_run/<file>.json to merge_dir/per_run/<file>__seed<N>.json
    so the existing analysis functions (which group by 'seed' field in JSON)
    see all seeds together."""
    merge_per_run = merge_dir / "per_run"
    merge_per_run.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for seed, d in seed_dirs:
        src_dir = d / stage_subdir / "per_run"
        if not src_dir.exists():
            logger.warning(f"  missing {src_dir}")
            continue
        for src in src_dir.glob("*.json"):
            # Each per-run JSON already contains 'seed' field. Suffix filename
            # so a unique flat layout is presented to the aggregator.
            stem = src.stem
            dst = merge_per_run / f"{stem}__seed{seed}.json"
            shutil.copyfile(src, dst)
            n_total += 1
    logger.info(f"  [{stage_subdir}] materialized {n_total} per-run JSONs from "
                f"{len(seed_dirs)} seeds")
    return n_total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_dir", required=True,
                   help="Directory containing seed_<N>/ subdirs (e.g. "
                        "/srv/nfs/results/p18_threshold_sensitivity)")
    p.add_argument("--output_dir", required=True,
                   help="Aggregated output dir (e.g. .../aggregated)")
    p.add_argument("--require_all_seeds", type=int, default=10,
                   help="Refuse to aggregate unless this many done.json present "
                        "(default 10; pass 0 to allow partial)")
    return p.parse_args()


def main():
    args = parse_args()
    seeds_dir = Path(args.seeds_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("aggregate", output_dir / "aggregate.log")

    logger.info("=" * 60)
    logger.info(f"P18 aggregate.py | RUN_TAG={RUN_TAG}")
    logger.info(f"  seeds_dir   = {seeds_dir}")
    logger.info(f"  output_dir  = {output_dir}")
    logger.info("=" * 60)

    seed_dirs = discover_seeds(seeds_dir)
    logger.info(f"Discovered {len(seed_dirs)} completed seeds: "
                f"{[s for s, _ in seed_dirs]}")
    if args.require_all_seeds > 0 and len(seed_dirs) < args.require_all_seeds:
        logger.error(f"Refusing to aggregate: only {len(seed_dirs)} of "
                     f"{args.require_all_seeds} seeds completed. "
                     f"Pass --require_all_seeds 0 to override.")
        sys.exit(2)

    # Distributions (seed-independent)
    collect_distributions(seed_dirs, output_dir / PATHS.distributions, logger)

    # Materialise per-run JSONs into flat layout that analysis.* expects.
    # We use a tempdir so we don't pollute output_dir with intermediate files,
    # but persist the final summary/stats/tables/figures to output_dir.
    with tempfile.TemporaryDirectory(prefix="p18_agg_", dir=output_dir) as td:
        td_path = Path(td)
        for stage_subdir, paths_attr in [
            ("cross_classifier", PATHS.cross_classifier),
            ("lgbm_fine",        PATHS.lgbm_fine),
            ("weighting",        PATHS.weighting),
            ("per_type",         PATHS.per_type),
        ]:
            stage_merge = td_path / paths_attr
            n = materialize_stage_per_run(seed_dirs, stage_subdir, stage_merge, logger)
            if n == 0:
                continue

        # Run the existing analysis pipeline against the merged tempdir.
        # analysis.main() looks at <root>/<stage>/per_run/ and writes
        # <root>/<stage>/summary.json + stats.json + <root>/tables/*.csv
        sys.argv = ["analysis.py", "--output_root", str(td_path)]
        analysis.main()

        # Move analysis outputs (everything analysis just wrote) into output_dir.
        for stage_subdir, paths_attr in [
            ("cross_classifier", PATHS.cross_classifier),
            ("lgbm_fine",        PATHS.lgbm_fine),
            ("weighting",        PATHS.weighting),
            ("per_type",         PATHS.per_type),
        ]:
            for fname in ("summary.json", "stats.json", "analysis.log"):
                src = td_path / paths_attr / fname
                if src.exists():
                    dst = output_dir / paths_attr / fname
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
        # Tables
        src_tables = td_path / PATHS.tables
        if src_tables.exists():
            dst_tables = output_dir / PATHS.tables
            dst_tables.mkdir(parents=True, exist_ok=True)
            for csv in src_tables.glob("*.csv"):
                shutil.copyfile(csv, dst_tables / csv.name)
            logger.info(f"  tables -> {dst_tables}")

    # Figures (run against the now-final output_dir which has summary.json files).
    sys.argv = ["figures.py", "--output_root", str(output_dir)]
    figures.main()

    logger.info("=" * 60)
    logger.info(f"[AGGREGATE DONE] outputs in {output_dir}")
    logger.info(f"  - distributions/snapshot.json")
    logger.info(f"  - cross_classifier/summary.json + stats.json")
    logger.info(f"  - lgbm_fine/summary.json + stats.json")
    logger.info(f"  - weighting/summary.json + stats.json")
    logger.info(f"  - per_type/summary.json")
    logger.info(f"  - tables/*.csv")
    logger.info(f"  - figures/fig{{1..4}}.pdf+png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
