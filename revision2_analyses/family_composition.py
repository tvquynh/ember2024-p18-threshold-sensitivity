#!/usr/bin/env python3
"""Family composition of the retained malware across the threshold sweep.

Reviewer 1 asked (R1-C4) how the "malware class distribution" changes across
thresholds. Section 6.2 reports the class balance and the file-type mixture;
this script adds the reading a malware researcher is most likely to intend --
the distribution over malware families -- and asks the question that matters
for the paper's interpretation: does raising T change WHICH families the
classifier sees, or only how many samples of the same families it sees?

Reads four metadata columns only; no features, no models, no retraining.

Usage:
    python family_composition.py [--parquet DIR] [--out family_composition.json]
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
import polars as pl

PE = ["win32", "win64", "dot_net"]
GRID = [0.065, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
BAND = 0.05          # near-boundary band [T, T+BAND), as in Section 6.2
TOPN = 10


def herfindahl(shares: np.ndarray) -> float:
    """Sum of squared shares: 1.0 = one family, ~0 = perfectly even."""
    return float((shares ** 2).sum())


def normalized_entropy(shares: np.ndarray) -> float:
    """0 = one family dominates, 1 = uniform over the observed families."""
    s = shares[shares > 0]
    if s.size <= 1:
        return 0.0
    return float(-(s * np.log(s)).sum() / np.log(s.size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default=os.environ.get("EMBER2024_PARQUET_DIR", "data/ember2024"),
        help="Directory holding ember2024_train.parquet from the EMBER2024 "
             "release; override with --parquet or EMBER2024_PARQUET_DIR.")
    ap.add_argument("--out", default="family_composition.json")
    a = ap.parse_args()

    src = Path(a.parquet) / "ember2024_train.parquet"
    print(f"[load] {src} (4 metadata columns)")
    df = (pl.scan_parquet(src)
            .filter(pl.col("label") == 1)
            .filter(pl.col("file_type").str.to_lowercase().is_in(PE))
            .select(["family", "family_confidence", "detection_ratio"])
            .collect())
    print(f"[load] {df.height:,} PE training malware")

    fam = df["family"].fill_null("<unlabeled>").to_numpy()
    conf = df["family_confidence"].fill_null(0.0).to_numpy()
    r = df["detection_ratio"].to_numpy()

    unl = float((fam == "<unlabeled>").mean())
    print(f"[info] unlabeled family share: {unl*100:.2f}%")
    print(f"[info] distinct families: {len(set(fam)):,}")
    print(f"[info] family_confidence: min {conf.min():.3f} "
          f"median {np.median(conf):.3f} max {conf.max():.3f}")

    # Baseline top families, so every threshold is reported on the same axis.
    base = fam[r >= GRID[0]]
    names, counts = np.unique(base, return_counts=True)
    order = np.argsort(-counts)
    top = [n for n in names[order] if n != "<unlabeled>"][:TOPN]
    print(f"[info] top {TOPN} families at baseline: {top}")

    rows = []
    for T in GRID:
        keep = fam[r >= T]
        n = keep.size
        nm, nc = np.unique(keep, return_counts=True)
        share = {k: int(v) for k, v in zip(nm, nc)}
        allshares = nc / n
        row = {
            "T": T,
            "n_retained": int(n),
            "n_families": int(len(nm)),
            "unlabeled_share": round(float(share.get("<unlabeled>", 0)) / n, 5),
            "herfindahl": round(herfindahl(allshares), 5),
            "normalized_entropy": round(normalized_entropy(allshares), 5),
            "top_shares": {t: round(share.get(t, 0) / n, 5) for t in top},
        }
        # Concentration among ATTRIBUTED families only. Lumping every
        # unattributed sample into one pseudo-family inflates any
        # concentration index, and unattributed samples are precisely the
        # low-consensus ones, so the naive index would measure attribution
        # coverage rather than family concentration.
        lab = keep[keep != "<unlabeled>"]
        if lab.size:
            _, lc = np.unique(lab, return_counts=True)
            row["herfindahl_attributed"] = round(herfindahl(lc / lab.size), 5)
            row["entropy_attributed"] = round(normalized_entropy(lc / lab.size), 5)

        # Same statistics restricted to the near-boundary band [T, T+BAND).
        band = fam[(r >= T) & (r < T + BAND)]
        if band.size:
            bn, bc = np.unique(band, return_counts=True)
            bs = bc / band.size
            row["band_n"] = int(band.size)
            row["band_n_families"] = int(len(bn))
            row["band_unlabeled_share"] = round(
                float((band == "<unlabeled>").mean()), 5)
            row["band_herfindahl_naive"] = round(herfindahl(bs), 5)
            bl = band[band != "<unlabeled>"]
            if bl.size:
                bln, blc = np.unique(bl, return_counts=True)
                row["band_n_families_attributed"] = int(len(bln))
                row["band_herfindahl_attributed"] = round(
                    herfindahl(blc / bl.size), 5)
            row["band_top_shares"] = {
                t: round(int(dict(zip(bn, bc)).get(t, 0)) / band.size, 5) for t in top}
        rows.append(row)

    print(f"\nRETAINED SET (concentration among attributed families only)")
    print(f"{'T':>6} {'retained':>11} {'families':>9} {'HHI_attr':>9} "
          f"{'H_attr':>8} {'unattrib':>9}")
    for x in rows:
        print(f"{x['T']:>6.3f} {x['n_retained']:>11,} {x['n_families']:>9,} "
              f"{x.get('herfindahl_attributed', float('nan')):>9.4f} "
              f"{x.get('entropy_attributed', float('nan')):>8.4f} "
              f"{x['unlabeled_share']*100:>8.2f}%")

    print(f"\nNEAR-BOUNDARY BAND [T, T+{BAND})")
    print(f"{'T':>6} {'band n':>9} {'families':>9} {'HHI_attr':>9} {'unattrib':>9}")
    for x in rows:
        print(f"{x['T']:>6.3f} {x.get('band_n', 0):>9,} "
              f"{x.get('band_n_families_attributed', 0):>9,} "
              f"{x.get('band_herfindahl_attributed', float('nan')):>9.4f} "
              f"{x.get('band_unlabeled_share', float('nan'))*100:>8.2f}%")

    print(f"\nTop-{TOPN} family shares of retained malware (%)")
    hdr = "  ".join(f"{t[:11]:>11}" for t in top)
    print(f"{'T':>6}  {hdr}")
    for x in rows:
        print(f"{x['T']:>6.3f}  " +
              "  ".join(f"{x['top_shares'][t]*100:>11.2f}" for t in top))

    b, e = rows[0], rows[-1]
    print(f"\nAcross the full sweep T={b['T']} -> {e['T']}:")
    print(f"  families present : {b['n_families']:,} -> {e['n_families']:,} "
          f"({100*(e['n_families']-b['n_families'])/b['n_families']:+.1f}%)")
    print(f"  concentration HHI: {b['herfindahl']:.4f} -> {e['herfindahl']:.4f}")
    print(f"  normalized entropy: {b['normalized_entropy']:.4f} -> {e['normalized_entropy']:.4f}")
    drift = max(abs(e["top_shares"][t] - b["top_shares"][t]) for t in top)
    print(f"  largest top-{TOPN} share drift: {drift*100:.2f} pp")

    Path(a.out).write_text(json.dumps(
        {"top_families": top, "band_width": BAND, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
