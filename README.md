# Threshold sensitivity and calibration effects in VirusTotal consensus labeling

**Paper**: *Threshold sensitivity and calibration effects in VirusTotal consensus labeling: a cross-classifier study on EMBER2024*
**Submitted to**: Journal of Information Security and Applications (Elsevier)
**Authors**: Trong-Thua Huynh (PTIT, first), Van-Quynh Trinh (PTIT, corresponding), De-Thu Huynh (SIU), Ngoc-Hieu Le (PTIT)
**Funding**: Posts and Telecommunications Institute of Technology (PTIT), Vietnam

This repository accompanies the manuscript and contains all source code,
configurations, and aggregated results needed to reproduce the 570 training
runs reported in the paper. The threshold grid was finalised at six coarse
points (`{0.065, 0.10, 0.15, 0.20, 0.30, 0.50}`) plus a five-point LightGBM
interior fine grid (`{0.08, 0.12, 0.18, 0.25, 0.40}`); we do not sweep below
the EMBER2024 default `T_base = 0.065` because the training-malware
distribution has only ≈1.2% of mass there.

---

## Quick start

### Option A — Single-node verification (smoke, ~1 hour)

Run the full pipeline at prototype scale (~20K training samples, 2 seeds)
on a single Linux box with 60+ CPU cores and 64+ GB RAM. Useful to verify
the code runs end-to-end before committing to the full study.

```bash
git clone <this-repo-url> p18_threshold_sensitivity
cd p18_threshold_sensitivity

# Python 3.10+, no GPU required
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# PyTorch CPU-only (saves ~2 GB):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Smoke test (~1 hour total)
python run_all.py --prototype --parquet_dir /path/to/EMBER2024/parquet
```

Expected outputs in `results/`:
- `distributions/snapshot.json` — detection-ratio statistics (median 0.684 train, 0.145 challenge)
- `cross_classifier/per_run/*.json` — 72 model results (4 clf × 9 T × 2 seeds)
- `lgbm_fine/per_run/*.json` — 28 results
- `weighting/per_run/*.json` — 20 results
- `per_type/per_run/*.json` — 24 results
- `figures/fig{1..4}.{pdf,png}` — 4 publication figures
- `tables/*.csv` — publication-ready tables

### Option B — Cluster reproduction (full study, ~11 hours)

The paper reports results from a 10-node SLURM cluster (master 128c +
9 compute 60c each, 668 vCPUs, 2.55 TB RAM total). To reproduce on a
similar cluster:

```bash
# 1. Stage code on shared NFS
scp -r ./* master:/srv/nfs/code/p18_threshold_sensitivity/

# 2. Stage EMBER2024 parquet on each node (local SSD recommended to avoid NFS contention):
#    <PARQUET_DIR>/ember2024_{train,test,challenge}.parquet

# 3. Submit the array
ssh master
cd /srv/nfs/code/p18_threshold_sensitivity
bash submit_array.sh

# 4. Aggregate when all 10 seeds done
python aggregate.py \
    --seeds_dir /srv/nfs/results/p18_threshold_sensitivity \
    --output_dir /srv/nfs/results/p18_threshold_sensitivity/aggregated
```

The full pipeline trains **570 PE classifiers**:
- 240 cross-classifier (4 clf × 6 T × 10 seeds)
- 110 LightGBM fine grid (1 × 11 T × 10 seeds)
- 100 detection-weighted (5 LightGBM + 3 XGBoost non-uniform + uniform baseline per family = 10 strats × 10 seeds)
- 120 per-file-type (3 ft × 4 T × 10 seeds)

Wall-clock ≈ 11 hours on the reference cluster (slowest seed dominates).

### Option C — Inspect aggregated results without re-running

`results_aggregated/` contains the final outputs from our run, ready to inspect:

```bash
cat results_aggregated/cross_classifier/summary.json | python -m json.tool | less
cat results_aggregated/cross_classifier/stats.json | python -m json.tool | less
ls results_aggregated/figures/    # 4 PDFs (the paper figures)
ls results_aggregated/tables/     # 10 CSVs (the paper tables)
```

The CSVs are the same numbers used in the paper's tables.

---

## Repository contents

```
.
├── README.md                              ← this file (reviewer-facing)
├── RUNBOOK.md                             ← 9-section operational runbook
├── LICENSE                                ← MIT License
├── requirements.txt                       ← Python 3.10+ pinned dependencies
│
│   ── Cluster + single-node entry points ──
├── run_all.py                             ← single-node entry (legacy, all seeds in one process)
├── run_seed.py                            ← cluster worker (1 seed runs all 4 stages)
├── run_seed.sbatch                        ← SLURM wrapper for run_seed.py
├── submit_array.sh                        ← cluster launcher (10 sbatch jobs)
├── aggregate.py                           ← merge 10-seed results to final outputs
│
│   ── Library modules ──
├── config.py                              ← thresholds, classifier params, seeds (FROZEN)
├── ember_v3_schema.py                     ← EMBER2024 v3 feature schema (2,568 dims)
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
│   ├── _common.py                         ← logging / atomic JSON / resume helpers
│   ├── run_distributions.py               ← detection-ratio snapshot
│   ├── run_cross_classifier.py            ← Exp 1: 4 clf × 9 T × 10 seeds
│   ├── run_lgbm_finegrid.py               ← Exp 2: LGB × 14 T × 10 seeds
│   ├── run_weighting.py                   ← Exp 3: 6 LGB strats + 4 XGB strats
│   └── run_per_type.py                    ← Exp 4: LGB per win32/win64/.NET
│
│   ── Tests ──
├── tests/
│   └── test_smoke.py                      ← 23 unit tests (schema/filter/weights/stats/calib)
│
│   ── Aggregated reference outputs ──
├── results_aggregated/                    ← outputs from the run reported in the paper
│   ├── distributions/snapshot.json
│   ├── cross_classifier/{summary,stats}.json
│   ├── lgbm_fine/{summary,stats}.json
│   ├── weighting/{summary,stats}.json
│   ├── per_type/summary.json
│   ├── tables/*.csv                       ← publication-ready CSVs
│   └── figures/fig{1..4}.{pdf,png}        ← 4 paper figures
│
├── deploy_below_baseline.sh               ← deployment helper script
└── .gitignore
```

---

## Dataset

**EMBER2024** is publicly available from the dataset authors:

> Joyce, R. J., Miller, B., Roth, N., Zak, A., Zaresky-Williams, A.,
> Anderson, H. S., Raff, E., & Holt, J. (2025). EMBER2024 — A Benchmark
> Dataset for Holistic Evaluation of Malware Classifiers. In KDD'25.
> https://github.com/FutureComputing4AI/EMBER2024

Our pipeline expects three Parquet files in a single directory:
- `ember2024_train.parquet` (2.34M samples, 2,587 columns)
- `ember2024_test.parquet` (540K samples)
- `ember2024_challenge.parquet` (4,868 evasive malware)

Each row has:
- `sha256`, `md5`, `tlsh` (identifiers)
- `file_type`, `label`, `family` (categorical)
- `detection_ratio` (the variable our paper studies)
- `feature_0000` … `feature_2567` (2,568 numeric features)
- 7 categorical feature indices `[2, 3, 4, 5, 6, 701, 702]` — passed to
  LightGBM as `categorical_feature`, per the EMBER2024 reference
  implementation (`thrember/model.py` from the dataset repo).

We do not redistribute EMBER2024 in this repository; please obtain it from
the dataset authors at the link above.

---

## Reproducibility contract

- **10 fixed random seeds**: `[42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]`
- **All hyperparameters frozen** in `config.py` (`RUN_TAG = jisa_v2_2026_04`).
- **Atomic writes**: per-run JSON files are written via temp + rename, so
  `--resume` reliably skips any (classifier, threshold, seed) combination
  whose result file already exists.
- **CPU-only throughout** (no GPU). LightGBM and XGBoost use `n_jobs`
  matched to the SLURM CPU allocation (120 on the master, 56 on compute
  nodes); the MLP uses PyTorch CPU with `torch.set_num_threads` matched
  to the same allocation. Therefore GPU non-determinism is not a source
  of variance.
- **Statistical tests** (Wilcoxon signed-rank, Cohen's d, Friedman,
  Holm-Bonferroni at α=0.05) are implemented in `stats.py` against
  `scipy.stats`. The exact computations and seed-paired structures are
  documented in the paper's Section 3.4.

If a re-run on the reference cluster does not match our reported numbers
within sampling noise, please open an issue — we will investigate.

---

## Test the install

```bash
pytest tests/ -q
# expected: 23 passed
```

The unit tests cover schema validation, threshold filtering, weighting
strategies, calibration metrics (ECE/MCE/Brier on synthetic data), and
multi-FPR computation. They run in <2 seconds on a laptop and do not
require the EMBER2024 dataset.

---

## Citation

```bibtex
@article{trinh2026threshold,
  author    = {Trinh, Van-Quynh and Huynh, Trong-Thua and Huynh, De-Thu and Le, Ngoc-Hieu},
  title     = {Threshold Sensitivity and Calibration Effects in {VirusTotal} Consensus Labeling: A Cross-Classifier Study on {EMBER2024}},
  journal   = {Journal of Information Security and Applications},
  year      = {2026},
  note      = {under review}
}
```

A Zenodo DOI will be minted at acceptance and added here.

---

## License

Released under the MIT License (see `LICENSE` in this repository).
The EMBER2024 dataset is licensed separately by its authors; please consult
the dataset's repository for its licensing terms.

---

## Contact

- **First author**: Trong-Thua Huynh (`thuaht@ptit.edu.vn`)
- **Corresponding author**: Van-Quynh Trinh (`quynhtv@ptithcm.edu.vn`)
- ORCID: 0000-0003-3934-1067 (T.-T. Huynh) / 0009-0006-0514-6123 (V.-Q. Trinh) 
- Affiliation: Posts and Telecommunications Institute of Technology, Ho Chi Minh City, Vietnam
