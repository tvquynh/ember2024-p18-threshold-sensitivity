# ********* *********** *** *********** ******* ** ********** ********* ********

**Paper title (redacted)**: ********* *********** *** *********** ******* ** ********** ********* ********: * **************** ***** ** *********
**Submitted to**: a peer-reviewed journal (revision under review)
**Release**: `jisa-revision-1` — the frozen state accompanying the first
revision. The paper title, author list, and dataset name are redacted
while the manuscript is under peer review; they will be restored on
acceptance (see `.restore-on-accept/`).

This repository accompanies the manuscript and contains all source code,
configurations, and aggregated results needed to reproduce the 570
training runs reported in the paper. The threshold grid is six coarse
points (`{0.065, 0.10, 0.15, 0.20, 0.30, 0.50}`) plus a five-point
interior fine grid (`{0.08, 0.12, 0.18, 0.25, 0.40}`), i.e. 11 fine-grid
thresholds in total; we do not sweep below the default `T_base = 0.065`
because the training-malware distribution has only ≈1.3% of its mass
there.

## What is in this release

| Path | Contents |
|---|---|
| `results_aggregated/*/summary.json` | Per-threshold aggregates **including the per-seed value arrays** behind every table and figure |
| `results_aggregated/figures/fig1..fig4` | Publication figures 1–4 |
| `revision1_analyses/figures/fig5,fig6` | Publication figures 5–6, added in this revision |
| `results_aggregated/tables/*.csv`, `revision1_analyses/tables/*.csv` | Supporting tables |
| `revision1_analyses/recompute_paper_stats.py` | **Authoritative** regeneration of every inferential number in the paper |
| `revision1_analyses/recomputed_stats_revision1.json` | Its output |
| `revision1_analyses/mechanism_analysis.py` | Retained-set composition diagnostics (figure 6) |
| `revision1_analyses/calibration_sensitivity.py` | Calibration-robustness analysis (figure 5) |
| `weights.py` | The eight detection-weighting formulas exactly as tabulated in the paper |

### Verifying the paper's numbers in one command

```bash
python revision1_analyses/recompute_paper_stats.py
```

This reads only the stored per-seed arrays and reproduces the Friedman
statistics, every Wilcoxon exact p-value, the Holm-Bonferroni adjustments
at the family sizes the paper states, Cohen's *d*, and the Random Forest
rank correlation. No model is retrained and no raw dataset is required.

> **Provenance note on `results_aggregated/*/stats.json`.** Those files
> come from an earlier aggregation pass over a larger threshold grid
> (9 coarse / 14 fine, including thresholds below the baseline) that the
> study later dropped. They are kept for provenance only, and their
> `correction`, `family_size` and `k` fields do **not** describe the
> analysis reported in the paper. Use
> `revision1_analyses/recomputed_stats_revision1.json` instead.

---

## Quick start

### Option A — Single-node verification (smoke, ~1 hour)

Run the full pipeline at prototype scale (~20K training samples, 2 seeds)
on a single Linux box with 60+ CPU cores and 64+ GB RAM. Useful to verify
the code runs end-to-end before committing to the full study.

```bash
git clone <this-repo-url> threshold_sensitivity
cd threshold_sensitivity

# Python 3.10+, no GPU required
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# PyTorch CPU-only (saves ~2 GB):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Smoke test (~1 hour total)
python run_all.py --prototype --parquet_dir /path/to/dataset/parquet
```

Expected outputs in `results/`:
- `distributions/snapshot.json` — detection-ratio statistics
- `cross_classifier/per_run/*.json` — per-run results
- `lgbm_fine/per_run/*.json`
- `weighting/per_run/*.json`
- `per_type/per_run/*.json`
- `figures/fig{1..4}.{pdf,png}` — 4 publication figures
- `tables/*.csv` — publication-ready tables

### Option B — Cluster reproduction (full study, ~11 hours)

The paper reports results from a 10-node SLURM cluster (master 128c +
9 compute 60c each, 668 vCPUs, 2.55 TB RAM total). To reproduce on a
similar cluster:

```bash
# 1. Stage code on shared NFS
scp -r ./* master:/srv/nfs/code/threshold_sensitivity/

# 2. Stage parquet on each node (local SSD recommended to avoid NFS contention):
#    <PARQUET_DIR>/dataset_{train,test,challenge}.parquet

# 3. Submit the array
ssh master
cd /srv/nfs/code/threshold_sensitivity
bash submit_array.sh

# 4. Aggregate when all 10 seeds done
python aggregate.py \
    --seeds_dir /srv/nfs/results/threshold_sensitivity \
    --output_dir /srv/nfs/results/threshold_sensitivity/aggregated
```

The full pipeline trains **570 PE classifiers**:
- 240 cross-classifier (4 clf × 6 T × 10 seeds)
- 110 LightGBM fine grid (1 × 11 T × 10 seeds)
- 100 detection-weighted (5 LightGBM + 3 XGBoost non-uniform + uniform baseline per family)
- 120 per-file-type (3 ft × 4 T × 10 seeds)

Wall-clock ≈ 11 hours on the reference cluster (slowest seed dominates).

### Option C — Inspect aggregated results without re-running

`results_aggregated/` contains the final outputs from our run, ready to inspect:

```bash
cat results_aggregated/cross_classifier/summary.json | python -m json.tool | less
cat results_aggregated/cross_classifier/stats.json | python -m json.tool | less
ls results_aggregated/figures/
ls results_aggregated/tables/
```

The CSVs are the same numbers used in the paper's tables.

---

## Repository contents

```
.
├── README.md                              ← this file
├── RUNBOOK.md                             ← 9-section operational runbook
├── LICENSE                                ← MIT License
├── requirements.txt                       ← Python 3.10+ pinned dependencies
│
│   ── Cluster + single-node entry points ──
├── run_all.py                             ← single-node entry
├── run_seed.py                            ← cluster worker (1 seed runs all 4 stages)
├── run_seed.sbatch                        ← SLURM wrapper for run_seed.py
├── submit_array.sh                        ← cluster launcher (10 sbatch jobs)
├── aggregate.py                           ← merge 10-seed results to final outputs
│
│   ── Library modules ──
├── config.py                              ← thresholds, classifier params, seeds (FROZEN)
├── ember_v3_schema.py                     ← feature schema (2,568 dims)
├── data_loader.py                         ← polars-based parquet loader
├── classifiers.py                         ← uniform API for LGB/XGB/RF/MLP
├── weights.py                             ← 6 detection-weighting strategies
├── metrics.py                             ← TPR@FPR + ECE/MCE/Brier
├── stats.py                               ← Wilcoxon + Cohen's d + Friedman + Holm
├── analysis.py                            ← aggregate per-run JSON → summary/stats
├── figures.py                             ← 4 publication PDFs
│
│   ── Per-stage runners (used by run_seed.py) ──
├── runners/
│   ├── _common.py
│   ├── run_distributions.py
│   ├── run_cross_classifier.py
│   ├── run_lgbm_finegrid.py
│   ├── run_weighting.py
│   └── run_per_type.py
│
│   ── Tests ──
├── tests/
│   └── test_smoke.py                      ← 23 unit tests
│
│   ── Aggregated reference outputs ──
├── results_aggregated/
│   ├── distributions/snapshot.json
│   ├── cross_classifier/{summary,stats}.json
│   ├── lgbm_fine/{summary,stats}.json
│   ├── weighting/{summary,stats}.json
│   ├── per_type/summary.json
│   ├── tables/*.csv
│   └── figures/fig{1..4}.{pdf,png}
│
└── .gitignore
```

---

## Dataset

The dataset name and reference are redacted while the manuscript is under
peer review. Reviewers and editors have access to the full citation
through the manuscript's Data Availability section. The dataset is
publicly available; details will be added here upon paper acceptance.

Our pipeline expects three Parquet files in a single directory:
- `dataset_train.parquet`
- `dataset_test.parquet`
- `dataset_challenge.parquet` (evasive malware split)

We do not redistribute the dataset in this repository; please obtain it
from the original dataset authors (see manuscript references).

---

## Authorship

The author list is redacted while the manuscript is under peer review.
Full author information will be added here upon paper acceptance.

## License

MIT License — see `LICENSE` file.

---

## Restoration on paper acceptance

When the paper is accepted for publication, the following will be
restored from `.restore-on-accept/`:

- Full paper title
- Author list
- Dataset name and reference
- Funder acknowledgement

The backup of the original (pre-redaction) `README.md`, `CITATION.cff`,
and `RUNBOOK.md` is preserved in `.restore-on-accept/` for that purpose.
