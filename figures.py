#!/usr/bin/env python3
"""figures.py — Publication figures for the JISA manuscript.

Reads the aggregated JSONs produced by analysis.py and emits 3 PDFs:

    fig1_detection_ratio_distributions.pdf
        (a) density histograms train vs challenge malware
        (b) CDFs with T=0.10 / T=0.15 vertical lines

    fig2_cross_classifier_sensitivity.pdf
        (a) Challenge TPR@1%FPR ±1 std vs T for all 4 classifiers
            plus shaded sweet zone
        (b) Test TPR@1%FPR vs T (shows minimal degradation)

    fig3_variance_stabilization.pdf
        (a) LGBM challenge TPR@1%FPR with error bars; sweet-zone
            thresholds highlighted
        (b) Standard deviation vs T with reduction-factor annotations

All figures use Elsevier-friendly vector output (PDF), DPI 300 fallback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (PATHS, CLASSIFIER_COLORS, FIG_DPI, SWEET_ZONE, BASELINE_T,
                    COLOR_BASELINE, COLOR_SWEET, CLASSIFIER_NAMES, RUN_TAG)
from runners._common import setup_logger, read_json


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # TrueType embed (required by some publishers)
        "ps.fonttype": 42,
    })


# ── Figure 1: detection-ratio distributions ──────────────────────────────────
def figure_distributions(snapshot_path: Path, out_pdf: Path):
    import matplotlib.pyplot as plt
    _style()
    snap = read_json(snapshot_path)
    hist = snap["histogram"]
    edges = np.array(hist["bin_edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])

    train_h = np.array(hist["train_malware_ratios"], dtype=np.float64)
    ch_h = np.array(hist["challenge_malware_ratios"], dtype=np.float64)
    train_d = train_h / max(train_h.sum(), 1) / np.diff(edges)
    ch_d = ch_h / max(ch_h.sum(), 1) / np.diff(edges)
    train_cdf = np.cumsum(train_h) / max(train_h.sum(), 1)
    ch_cdf = np.cumsum(ch_h) / max(ch_h.sum(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

    ax = axes[0]
    ax.bar(centers, train_d, width=np.diff(edges)[0], alpha=0.55,
           label=f"Train malware (median {snap['global']['train_malware']['median']:.3f})",
           color="#FF7043", edgecolor="none")
    ax.bar(centers, ch_d, width=np.diff(edges)[0], alpha=0.55,
           label=f"Challenge malware (median {snap['global']['challenge_malware']['median']:.3f})",
           color="#1E88E5", edgecolor="none")
    ax.axvline(BASELINE_T, color=COLOR_BASELINE, lw=0.8, ls=":")
    ax.axvline(0.15, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Detection ratio $r$")
    ax.set_ylabel("Density")
    ax.set_title("(a) Density")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)

    ax = axes[1]
    ax.plot(centers, train_cdf, "-", color="#E65100", lw=1.6, label="Train malware")
    ax.plot(centers, ch_cdf,    "-", color="#0D47A1", lw=1.6, label="Challenge malware")
    ax.axvline(0.10, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0.15, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.10, 0.02, " T=0.10", fontsize=8, rotation=90, va="bottom")
    ax.text(0.15, 0.02, " T=0.15", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Detection ratio $r$")
    ax.set_ylabel("CDF")
    ax.set_title("(b) Cumulative distribution")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"))
    plt.close(fig)


# ── Figure 2: cross-classifier threshold sensitivity ─────────────────────────
def _series(summary_by_T: Dict[str, dict], split: str, metric: str):
    Ts, means, stds = [], [], []
    for T_str, bucket in sorted(summary_by_T.items(), key=lambda x: float(x[0])):
        cell = bucket.get(split, {}).get(metric)
        if cell is None:
            continue
        Ts.append(float(T_str))
        means.append(cell["mean"] * 100)
        stds.append(cell["std"] * 100)
    return np.array(Ts), np.array(means), np.array(stds)


def figure_cross_classifier(summary_path: Path, out_pdf: Path,
                            metric: str = "TPR@1.0%FPR"):
    import matplotlib.pyplot as plt
    _style()
    sum_all = read_json(summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))

    for ax, split, title in [
        (axes[0], "challenge", "(a) Challenge TPR@1%FPR"),
        (axes[1], "test",      "(b) Test TPR@1%FPR"),
    ]:
        ax.axvspan(SWEET_ZONE[0], SWEET_ZONE[1], color=COLOR_SWEET,
                   alpha=0.35, lw=0, zorder=0)
        for clf in CLASSIFIER_NAMES:
            if clf not in sum_all:
                continue
            Ts, means, stds = _series(sum_all[clf], split, metric)
            if len(Ts) == 0:
                continue
            color = CLASSIFIER_COLORS[clf]
            ax.errorbar(Ts, means, yerr=stds, fmt="o-", color=color,
                        label=clf, lw=1.4, ms=4, capsize=2,
                        alpha=0.95, zorder=3)
        ax.axvline(BASELINE_T, color=COLOR_BASELINE, lw=0.8, ls=":",
                   alpha=0.7, zorder=1)
        ax.set_xlabel("Threshold $T$")
        ax.set_ylabel("TPR@1%FPR (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"))
    plt.close(fig)


# ── Figure 3: variance stabilization (LGBM) ──────────────────────────────────
def figure_variance(summary_path_coarse: Path,
                    summary_path_fine: Path,
                    out_pdf: Path,
                    metric: str = "TPR@1.0%FPR"):
    import matplotlib.pyplot as plt
    _style()
    summaries = []
    for p in (summary_path_coarse, summary_path_fine):
        if p.exists():
            s = read_json(p)
            if "LightGBM" in s:
                summaries.append(s["LightGBM"])
    # Merge into one dict preferring fine grid when a T exists in both.
    merged: Dict[str, dict] = {}
    for s in summaries:
        merged.update({k: v for k, v in s.items() if k not in merged})
    merged.update(summaries[-1] if summaries else {})

    Ts, means, stds = _series(merged, "challenge", metric)
    if len(Ts) == 0:
        raise RuntimeError("No LightGBM data found for variance figure")

    # Baseline std for reduction factor.
    base_std = stds[np.argmin(np.abs(Ts - BASELINE_T))]
    reductions = base_std / np.clip(stds, 1e-9, None)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

    ax = axes[0]
    colors = np.where(
        (Ts >= SWEET_ZONE[0]) & (Ts <= SWEET_ZONE[1]),
        "#2C7BB6", "#C0C0C0"
    )
    ax.bar(Ts, means, yerr=stds, color=colors, width=0.014,
           capsize=3, edgecolor="black", lw=0.3, error_kw={"lw": 0.8})
    ax.axvline(BASELINE_T, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Threshold $T$")
    ax.set_ylabel("Challenge TPR@1%FPR (%)")
    ax.set_title("(a) Mean $\\pm$ 1 std (LightGBM, 10 seeds)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(Ts, stds, "o-", color="#D62728", lw=1.5, ms=5)
    ax.axvline(BASELINE_T, color="black", lw=0.8, ls=":", alpha=0.7)
    for T_i, s_i, r_i in zip(Ts, stds, reductions):
        if r_i > 2.0:
            ax.annotate(f"{r_i:.0f}×", xy=(T_i, s_i),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color="#333333")
    ax.set_xlabel("Threshold $T$")
    ax.set_ylabel("Std of challenge TPR@1%FPR (%)")
    ax.set_title("(b) Seed-to-seed variance")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.28)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"))
    plt.close(fig)


# ── Figure 4: Calibration reliability diagram (LightGBM × T sweep) ───────────
def figure_calibration(summary_path_fine: Path, out_pdf: Path,
                        thresholds_to_show=(0.065, 0.10, 0.20, 0.50),
                        split: str = "challenge"):
    """Reliability diagram for LGBM at selected T values + ECE bar chart."""
    import matplotlib.pyplot as plt
    _style()
    if not summary_path_fine.exists():
        return
    summary = read_json(summary_path_fine)
    if "LightGBM" not in summary:
        return
    lgb = summary["LightGBM"]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # (a) Reliability diagrams overlay
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
    cmap = plt.cm.viridis
    for i, T in enumerate(sorted(thresholds_to_show)):
        T_str = f"{T:.3f}"
        if T_str not in lgb:
            continue
        bucket = lgb[T_str].get(split, {})
        bins = bucket.get("reliability_bins", [])
        if not bins:
            continue
        xs = [b.get("mean_pred_avg", float("nan")) for b in bins if b.get("n_total", 0) > 0]
        ys = [b.get("frac_pos_avg", float("nan")) for b in bins if b.get("n_total", 0) > 0]
        if not xs:
            continue
        color = cmap(i / max(len(thresholds_to_show) - 1, 1))
        ece = bucket.get("ECE", {}).get("mean", float("nan"))
        ax.plot(xs, ys, "o-", color=color, ms=4, lw=1.2,
                label=f"T={T:.3f} (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"(a) Reliability diagram ({split} split, LightGBM)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (b) ECE across all T values
    ax = axes[1]
    Ts = sorted([float(k) for k in lgb.keys()])
    eces = []
    ece_stds = []
    for T in Ts:
        T_str = f"{T:.3f}"
        bucket = lgb[T_str].get(split, {})
        ece = bucket.get("ECE", {})
        eces.append(ece.get("mean", float("nan")))
        ece_stds.append(ece.get("std", 0.0))
    Ts_arr = np.array(Ts)
    eces_arr = np.array(eces)
    ece_stds_arr = np.array(ece_stds)
    in_sweet = (Ts_arr >= SWEET_ZONE[0]) & (Ts_arr <= SWEET_ZONE[1])
    ax.bar(Ts_arr[~in_sweet], eces_arr[~in_sweet], yerr=ece_stds_arr[~in_sweet],
           color="#C0C0C0", width=0.014, capsize=3, edgecolor="black", lw=0.3,
           error_kw={"lw": 0.6}, label="Outside sweet zone")
    ax.bar(Ts_arr[in_sweet], eces_arr[in_sweet], yerr=ece_stds_arr[in_sweet],
           color="#2C7BB6", width=0.014, capsize=3, edgecolor="black", lw=0.3,
           error_kw={"lw": 0.6}, label="Sweet zone")
    ax.axvline(BASELINE_T, color="black", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Threshold $T$")
    ax.set_ylabel("ECE (challenge split)")
    ax.set_title("(b) Expected Calibration Error vs T (LightGBM)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"))
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=PATHS.output_root)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_root)
    fig_dir = root / PATHS.figures
    logger = setup_logger("figures", root / "figures.log")
    logger.info(f"RUN_TAG={RUN_TAG}; root={root}")

    snap = root / PATHS.distributions / "snapshot.json"
    if snap.exists():
        figure_distributions(snap, fig_dir / "fig1_detection_ratio_distributions.pdf")
        logger.info("[fig1] distributions figure written")
    else:
        logger.warning(f"Skip fig1 — {snap} missing. Run run_distributions.py first.")

    cc = root / PATHS.cross_classifier / "summary.json"
    if cc.exists():
        figure_cross_classifier(cc, fig_dir / "fig2_cross_classifier_sensitivity.pdf")
        logger.info("[fig2] cross-classifier figure written")
    else:
        logger.warning(f"Skip fig2 — {cc} missing. Run analysis.py first.")

    fg = root / PATHS.lgbm_fine / "summary.json"
    if cc.exists():
        figure_variance(cc, fg, fig_dir / "fig3_variance_stabilization.pdf")
        logger.info("[fig3] variance figure written")

    if fg.exists():
        figure_calibration(fg, fig_dir / "fig4_calibration_reliability.pdf")
        logger.info("[fig4] calibration figure written")
    else:
        logger.warning(f"Skip fig4 — {fg} missing.")

    logger.info(f"[DONE] figures in {fig_dir}")


if __name__ == "__main__":
    main()
