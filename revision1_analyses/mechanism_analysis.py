"""R1-C4 mechanism analysis: compute 4 diagnostics × 11 fine-grid thresholds
from the raw EMBER2024 parquet + existing distributions/snapshot.json.

Outputs:
  mechanism_analysis.json           (raw numbers for the paper text)
  figures/fig5_mechanism.pdf/.png   (2x2 panel diagnostic)
  tables/mechanism_summary.csv      (6 T × 4 diagnostics for response letter)

Diagnostics computed at each T in the fine grid:
  (a) malware:benign class ratio of the RETAINED training set
  (b) file-type composition (Win32/Win64/.NET/other) among retained malware
  (c) detection-ratio distribution of retained malware (histogram overlay)
  (d) retained sample difficulty proxy:
      d1 = median detection ratio of retained malware
      d2 = fraction of retained malware in the near-boundary band [T, T+0.05]
      d3 = mean detection ratio of retained malware

Run:
    python mechanism_analysis.py
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REVISION_DIR = Path(__file__).resolve().parent
FIG_DIR = REVISION_DIR / "figures"
TABLE_DIR = REVISION_DIR / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

TRAIN_PARQUET = Path("E:/project_data/parquet_clean-week/ember2024_train.parquet")

FINE_T = [0.065, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
PE_TYPES = {"win32": "Win32", "win64": "Win64", "dot_net": ".NET"}
BLUE_PALETTE = ["#1F4E79", "#2E75B6", "#5B9BD5", "#9DC3E6"]

# Filter Windows PE only (Win32, Win64, .NET) — matches paper scope
PE_FILTER_STRINGS = {"win32", "win64", "dot_net", "Win32", "Win64", ".NET"}


def load_malware_pe() -> pl.DataFrame:
    """Load only Windows PE malware rows with (detection_ratio, file_type)."""
    print("[load] scanning train parquet (this can take ~1-2 min)…")
    df = (
        pl.scan_parquet(TRAIN_PARQUET)
        .filter(pl.col("label") == 1)
        .filter(pl.col("file_type").str.to_lowercase().is_in(list({s.lower() for s in PE_FILTER_STRINGS})))
        .select(["detection_ratio", "file_type"])
        .collect()
    )
    # normalise file_type strings
    df = df.with_columns(pl.col("file_type").str.to_lowercase().alias("ft_norm"))
    print(f"[load] retained {df.height:,} PE malware rows")
    print(f"[load] file-type breakdown: {df.group_by('ft_norm').len().sort('len', descending=True).to_dict(as_series=False)}")
    return df


def compute_diagnostics(mal_df: pl.DataFrame, n_benign_fixed: int) -> list[dict]:
    """For each T in FINE_T compute the 4 diagnostics."""
    out = []
    total_mal_at_t0 = mal_df.height
    for T in FINE_T:
        retained = mal_df.filter(pl.col("detection_ratio") >= T)
        n_retained = retained.height
        # (a) class ratio
        mal_ben_ratio = n_retained / n_benign_fixed
        # (b) file-type composition
        ft_counts = {}
        if n_retained:
            grouped = retained.group_by("ft_norm").len()
            for row in grouped.iter_rows(named=True):
                ft_norm = row["ft_norm"]
                # Map back to canonical name
                canonical = "win32" if "32" in ft_norm else ("win64" if "64" in ft_norm else ("dot_net" if "net" in ft_norm else ft_norm))
                ft_counts[canonical] = ft_counts.get(canonical, 0) + row["len"]
        ft_composition = {ft: (ft_counts.get(ft, 0) / n_retained if n_retained else 0.0) for ft in ["win32", "win64", "dot_net"]}
        # (c) detection-ratio distribution stats
        det_ratios = retained["detection_ratio"].to_numpy() if n_retained else np.array([])
        # (d) difficulty proxies
        d1_median_r = float(np.median(det_ratios)) if n_retained else 0.0
        d3_mean_r = float(det_ratios.mean()) if n_retained else 0.0
        band_hi = min(T + 0.05, 1.0)
        near_boundary = retained.filter((pl.col("detection_ratio") >= T) & (pl.col("detection_ratio") < band_hi)).height
        d2_near_boundary_frac = near_boundary / n_retained if n_retained else 0.0
        entry = {
            "T": T,
            "n_malware_retained": n_retained,
            "n_benign": n_benign_fixed,
            "mal_ben_ratio": round(mal_ben_ratio, 4),
            "ft_composition": {k: round(v, 4) for k, v in ft_composition.items()},
            "median_detection_ratio_retained": round(d1_median_r, 4),
            "mean_detection_ratio_retained": round(d3_mean_r, 4),
            "near_boundary_frac_[T, T+0.05)": round(d2_near_boundary_frac, 4),
        }
        # small histogram (10 bins over [T, 1])
        if n_retained:
            hist, edges = np.histogram(det_ratios, bins=10, range=(T, 1.0))
            entry["detection_ratio_histogram_10bins"] = {
                "edges": [round(float(e), 3) for e in edges.tolist()],
                "counts": [int(c) for c in hist.tolist()],
            }
        out.append(entry)
        print(f"[T={T:.3f}] retained={n_retained:>7} ratio={mal_ben_ratio:.3f} ft={ft_composition} median_r={d1_median_r:.3f} near_boundary={d2_near_boundary_frac:.3f}")
    return out


def make_figure_5(diagnostics: list[dict], mal_df: pl.DataFrame, out_pdf: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    T_arr = np.array([d["T"] for d in diagnostics])

    # Panel (a): class ratio + malware count
    ax = axes[0, 0]
    ratios = np.array([d["mal_ben_ratio"] for d in diagnostics])
    ax2 = ax.twinx()
    counts = np.array([d["n_malware_retained"] for d in diagnostics]) / 1000.0
    ax.plot(T_arr, ratios, "o-", color=BLUE_PALETTE[0], lw=1.8, ms=6, label="mal:ben ratio")
    ax2.plot(T_arr, counts, "s--", color=BLUE_PALETTE[2], lw=1.2, ms=4, alpha=0.7, label="mal count (thousands)")
    ax.set_xlabel("Threshold $T$")
    ax.set_ylabel("Retained malware : benign ratio", color=BLUE_PALETTE[0])
    ax2.set_ylabel("Retained malware (thousands)", color=BLUE_PALETTE[2])
    ax.set_title("(a) Class balance of the retained training set")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.065, color="black", lw=0.8, ls=":", alpha=0.6)

    # Panel (b): file-type composition stacked area
    ax = axes[0, 1]
    win32 = np.array([d["ft_composition"]["win32"] for d in diagnostics])
    win64 = np.array([d["ft_composition"]["win64"] for d in diagnostics])
    dotnet = np.array([d["ft_composition"]["dot_net"] for d in diagnostics])
    ax.stackplot(T_arr, win32, win64, dotnet,
                 labels=["Win32", "Win64", ".NET"],
                 colors=BLUE_PALETTE[:3], alpha=0.85)
    ax.set_xlabel("Threshold $T$")
    ax.set_ylabel("Composition of retained malware")
    ax.set_title("(b) File-type composition")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.065, color="black", lw=0.8, ls=":", alpha=0.6)

    # Panel (c): detection-ratio histogram overlay for 4 representative T
    ax = axes[1, 0]
    representative_T = [0.065, 0.12, 0.20, 0.50]
    for i, T in enumerate(representative_T):
        r = mal_df.filter(pl.col("detection_ratio") >= T)["detection_ratio"].to_numpy()
        if len(r):
            hist, edges = np.histogram(r, bins=30, range=(0, 1), density=True)
            centers = 0.5 * (edges[1:] + edges[:-1])
            ax.plot(centers, hist, "-", color=BLUE_PALETTE[i], lw=1.5,
                    label=f"$T = {T}$ ($n={len(r):,}$)")
    ax.set_xlabel("Detection ratio $r$")
    ax.set_ylabel("Density (retained malware)")
    ax.set_title("(c) Detection-ratio distribution shift")
    # The distributions peak near r = 0.85, so an upper-right legend would sit
    # on top of the peak. Park it upper-left, over the empty mid-range, and add
    # headroom so it never touches the curves.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92,
              borderpad=0.4, labelspacing=0.3, handlelength=1.6)
    ax.grid(True, alpha=0.3)

    # Panel (d): difficulty proxies
    ax = axes[1, 1]
    median_r = np.array([d["median_detection_ratio_retained"] for d in diagnostics])
    mean_r = np.array([d["mean_detection_ratio_retained"] for d in diagnostics])
    near_boundary = np.array([d["near_boundary_frac_[T, T+0.05)"] for d in diagnostics])
    ax.plot(T_arr, median_r, "o-", color=BLUE_PALETTE[0], lw=1.5, ms=5, label="median $r$ of retained")
    ax.plot(T_arr, mean_r, "s-", color=BLUE_PALETTE[2], lw=1.5, ms=5, label="mean $r$ of retained")
    ax2 = ax.twinx()
    ax2.plot(T_arr, near_boundary, "^--", color="#C0504D", lw=1.2, ms=5, alpha=0.7,
             label=r"near-boundary fraction $[T, T{+}0.05)$")
    ax.set_xlabel("Threshold $T$")
    ax.set_ylabel("Detection ratio of retained set", color=BLUE_PALETTE[0])
    ax2.set_ylabel("Near-boundary fraction", color="#C0504D")
    ax.set_title("(d) Retained-sample difficulty proxies")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.065, color="black", lw=0.8, ls=":", alpha=0.6)
    # Both retained-ratio curves rise to the top right, so the lower-right
    # corner is the only region no curve crosses.
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.02)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92,
              borderpad=0.4, labelspacing=0.3, handlelength=1.6)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[figure5] wrote {out_pdf}")


def write_summary_table(diagnostics: list[dict], out_csv: Path):
    import csv
    rows = []
    for d in diagnostics:
        rows.append({
            "T": d["T"],
            "malware_retained": d["n_malware_retained"],
            "mal_ben_ratio": d["mal_ben_ratio"],
            "win32_pct": d["ft_composition"]["win32"],
            "win64_pct": d["ft_composition"]["win64"],
            "dotnet_pct": d["ft_composition"]["dot_net"],
            "median_r": d["median_detection_ratio_retained"],
            "mean_r": d["mean_detection_ratio_retained"],
            "near_boundary_frac": d["near_boundary_frac_[T, T+0.05)"],
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[table] wrote {out_csv}")


def main():
    n_benign_fixed = 1_170_000  # per paper Section 4.1 (all benign retained at every T)
    mal = load_malware_pe()
    diagnostics = compute_diagnostics(mal, n_benign_fixed)
    (REVISION_DIR / "mechanism_analysis.json").write_text(
        json.dumps({"run_tag": "revision1", "diagnostics": diagnostics}, indent=2),
        encoding="utf-8",
    )
    print(f"[json] wrote {REVISION_DIR / 'mechanism_analysis.json'}")
    make_figure_5(diagnostics, mal, FIG_DIR / "fig5_mechanism.pdf")
    write_summary_table(diagnostics, TABLE_DIR / "mechanism_summary.csv")


if __name__ == "__main__":
    main()
