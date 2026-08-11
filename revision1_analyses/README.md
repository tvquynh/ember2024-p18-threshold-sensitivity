# revision1_analyses/

Analyses added for the JISA Major Revision (2026-07-08 decision,
2026-09-01 deadline). These scripts reproduce the two new figures
added to the revised manuscript **without** re-running the full
training sweep — they read the pooled per-seed reliability bins and
per-run summaries produced by the original submission, plus a
one-time scan of the raw dataset parquet for the mechanism figure.

## Files

| File | Purpose |
|---|---|
| `mechanism_analysis.py` | Computes the four mechanism diagnostics across the 11 fine-grid thresholds and generates the mechanism panel (paper Figure 6 in the revised manuscript). Reads the raw dataset parquet for file-type composition; the pooled per-run summary for the other three panels. |
| `calibration_sensitivity.py` | Computes the ECE robustness re-analysis (equal-width, 5- and 15-quantile adaptive-binning, class-frequency-adjusted) and Brier-score decomposition from the existing pooled `reliability_bins` — no re-training. Produces paper Figure 5 in the revised manuscript. |
| `mechanism_analysis.json` | Numbers used in the mechanism paragraphs (Section 6.2 of the revised manuscript). |
| `calibration_sensitivity.json` | Numbers used in the calibration-robustness paragraph (Section 5.5). |
| `figures/` | PDF and PNG versions of the two new figures. |
| `tables/` | CSV summaries suitable for spot-checking against the paper text. |

## Reproducing

```bash
# From the repository root (venv activated):
python revision1_analyses/mechanism_analysis.py
python revision1_analyses/calibration_sensitivity.py
```

Both scripts write outputs into their respective subfolders and print a
per-threshold trace to stdout for quick inspection. Neither script
requires GPUs or the training pipeline; the mechanism script needs the
raw dataset parquet at the path set in the script (edit the constant if
your local path differs).

## Notes

- The mechanism script reports a U-shaped near-boundary fraction with
  interior minimum at $T = 0.20$ that supports the sweet-zone
  interpretation in the revised manuscript.
- The calibration-sensitivity script confirms that the sweet-zone
  ranking is preserved under three alternative binning strategies and
  under a class-frequency-adjusted variant that partially controls for
  the 1:55 imbalance of the challenge mixed split.
- Strict class-conditional ECE (calibration among positives only) would
  require per-sample per-class predictions; those are not stored in the
  current release. A commitment to store them in the next release is
  made in the revised manuscript.
