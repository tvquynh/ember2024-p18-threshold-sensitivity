# RUNBOOK — P18 Threshold Sensitivity (JISA)

Operational runbook for running the full experiment on the 73-core CPU server.
Self-contained: assumes the paper folder is checked out at
`E:\phase3\scripts\papers\p18_threshold_sensitivity\` and data lives at
`E:\project_data\parquet_clean-week\`.

---

## 1. Pre-flight (≤ 2 min)

Purpose: confirm the environment is sane *before* committing 40 h of compute.

```bash
# From the paper folder
cd E:/phase3/scripts/papers/p18_threshold_sensitivity

# 1a. Python + libs
python -c "import numpy, scipy, polars, sklearn, lightgbm, xgboost; print('libs OK')"
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"

# 1b. Data presence
python -c "
import polars as pl
for s in ('train','test','challenge'):
    df = pl.scan_parquet(f'E:/project_data/parquet_clean-week/ember2024_{s}.parquet').head(1).collect()
    print(s, '->', df.shape, '| cols with detection_ratio?', 'detection_ratio' in df.columns)
"

# 1c. Schema self-test
python ember_v3_schema.py

# 1d. Unit tests
pytest tests/ -q
```

**Expected**: all 4 commands exit cleanly. If `detection_ratio` column is missing, STOP — regenerate the parquet pipeline.

---

## 2. Smoke run (~10 min)

Purpose: end-to-end rehearsal with ~20K training samples and 2 seeds. Catches schema / API drift without burning server time.

```bash
python run_all.py --prototype
```

Verify:

- `results/distributions/snapshot.json` contains `global.train_malware.median ~ 0.68` (full data; prototype doesn't subset here).
- `results/cross_classifier/per_run/` has 48 json files (4 clf × 6 T × 2 seeds).
- `results/cross_classifier/summary.json` parses without errors.
- Figures directory has 3 PDFs.

If any of the above is missing, check `results/<stage>/run.log` — each runner is fail-isolated so other stages can still succeed.

---

## 3. Full run — production (~55 h)

```bash
# Backgrounded, with logs to file. Use screen / tmux in practice.
nohup python run_all.py \
    --parquet_dir <PARQUET_DIR> \
    --output_root results \
  > results/full_run.log 2>&1 &

echo $! > results/full_run.pid
```

Time budget per stage (reference timings on a 70-core / 400 GB-RAM CPU server):

| Stage                | Models | Wall clock |
|----------------------|-------:|-----------:|
| distributions        |      0 |    ~2 min  |
| cross_classifier     |    360 |   ~27 h    |
| lgbm_fine            |    140 |   ~13 h    |
| weighting            |    100 |    ~10 h   |
| per_type             |    120 |    ~4 h    |
| analysis + figures   |      — |    ~1 min  |

**Total**: ~54-58 h on a 70-core CPU server (verified empirically with first 5 LightGBM runs at ~7.5 min/model — see `results/cross_classifier/run.log`).

**Note**: cross_classifier model count includes the below-baseline thresholds (T ∈ {0.0, 0.02, 0.04}) added as a reviewer-anticipated control study.

### Resume after partial completion

If a previous full run completed only `--thresholds [0.065, 0.10, 0.15, 0.20, 0.30, 0.50]` (the pre-2026-04-25 grid), use `--resume` to add only the missing below-baseline runs:

```bash
nohup python run_all.py --resume \
    --parquet_dir <PARQUET_DIR> \
  > results/below_baseline_run.log 2>&1 &
```

Existing per-run JSON files are skipped automatically. Estimated incremental cost: **~18 h** (120 cross_classifier + 30 lgbm_fine new models).

---

## 4. Live monitoring

```bash
# Overall progress
tail -f results/run_all.out

# Per-stage detail
tail -f results/cross_classifier/run.log
tail -f results/lgbm_fine/run.log

# Count completed runs mid-flight
for d in cross_classifier lgbm_fine weighting per_type; do
    n=$(ls results/$d/per_run/*.json 2>/dev/null | wc -l)
    echo "$d: $n"
done

# Machine health
top -b -n 1 | head -5
free -h
df -h E:
```

---

## 5. Verify completeness (post-run)

```bash
# Count of per-run JSONs — each must match the plan
python -c "
from pathlib import Path
targets = {'cross_classifier': 240, 'lgbm_fine': 110,
           'weighting': 100, 'per_type': 120}
for d, n in targets.items():
    p = Path(f'results/{d}/per_run')
    got = len(list(p.glob('*.json'))) if p.exists() else 0
    status = 'OK' if got == n else 'MISSING'
    print(f'{d:20s} got={got:4d} expected={n:4d} [{status}]')
"
```

Any missing run: check the stage log for `[FAIL]` entries, then `python run_all.py --only <stage> --resume`.

---

## 6. Analysis — numbers for the paper (≤ 1 min)

```bash
# Re-aggregate (safe to re-run; no training)
python run_all.py --only analysis

# Numbers used in manuscript Tables 2-5 now live at:
#   results/cross_classifier/summary.json
#   results/cross_classifier/stats.json
#   results/lgbm_fine/summary.json
#   results/weighting/summary.json
#   results/per_type/summary.json

# Publication-ready CSV tables:
ls results/tables/
```

Quick sanity queries:

```bash
# LGBM challenge TPR@1% by T
python -c "
import json
s = json.load(open('results/cross_classifier/summary.json'))
for T, b in sorted(s['LightGBM'].items(), key=lambda x: float(x[0])):
    m = b['challenge']['TPR@1.0%FPR']
    print(f'T={T} mean={m[\"mean\"]*100:.2f}% std={m[\"std\"]*100:.2f}%')
"

# Wilcoxon significance
python -c "
import json
st = json.load(open('results/cross_classifier/stats.json'))
for clf, x in st['per_classifier'].items():
    pw = x.get('pairwise', {}).get('per_threshold', {})
    for T, r in pw.items():
        if r.get('significant_corrected'):
            print(f'{clf} T={T} p={r[\"p_value\"]:.4f} d={r[\"cohens_d\"]:.2f}')
"
```

### Red flags

- LGBM baseline `T=0.065` challenge TPR@1% < 0.50 or > 0.60 → check data or seed list.
- LGBM `T=0.10` mean < `T=0.065` mean → pipeline is inverted (most likely a threshold filter bug).
- XGBoost std `< 1e-6` → XGB_PARAMS subsample/colsample didn't take effect (check classifier config).
- MLP challenge TPR@1% < 0.30 → probably didn't get a val split; check `stratified_val_split`.

---

## 7. Figures → manuscript (seconds)

```bash
python run_all.py --only figures

# Copy into the LaTeX package
cp results/figures/fig1_detection_ratio_distributions.pdf   <manuscript>/
cp results/figures/fig2_cross_classifier_sensitivity.pdf    <manuscript>/
cp results/figures/fig3_variance_stabilization.pdf          <manuscript>/
```

Each PDF has a matching `.png` for quick visual inspection without a PDF viewer.

---

## 8. Backup (immediately after full run)

```bash
TS=$(date +%Y%m%d_%H%M)
tar -czf results_p18_$TS.tar.gz results/
# copy to OneDrive manuscript folder for reviewer reproducibility:
cp results_p18_$TS.tar.gz "<OneDrive paper folder>/data_archive/"
```

Archive size expected: 30–60 MB (JSON + PDFs + logs).

---

## 9. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `FileNotFoundError` on parquet | wrong `--parquet_dir` | check path, re-run with correct `--parquet_dir` |
| LightGBM "No further splits with positive gain" spam | `verbose` not set | already muted via `verbose=-1`; safe to ignore if rare |
| MemoryError during data load | concurrent heavy job | `free -h`, lower priority, or reduce `file_types` |
| Stage crashes mid-run | transient OS / IO | `python run_all.py --only <stage> --resume` — existing per-run JSON is reused |
| Tests in `tests/test_smoke.py` fail after upstream schema change | schema offsets moved | update `ember_v3_schema.py` in lockstep with upstream, then re-run tests |
| `sklearn RandomForest` memory spike | large `max_depth` + full data | already capped at `max_depth=20` + `min_samples_leaf=100`; should stay under 50 GB |
| XGBoost still shows `std≈0` | hist method override | verify `XGB_PARAMS["subsample"] == 0.8` (not `None` or `1.0`) |

---

## Hand-off

After a clean full run, the artifacts required by the manuscript / reviewers are:

- `results/*/summary.json` — all numbers for Tables 2-5.
- `results/*/stats.json`   — Wilcoxon / Friedman / Cohen d.
- `results/tables/*.csv`   — publication-ready tables.
- `results/figures/*.pdf`  — Figures 1-3 for the LaTeX package.
- `results/*/run.log`      — full audit trail.

Commit these under source control (or archive to OneDrive under the paper folder) *before* edits to `config.py` to preserve the results that map to the submitted manuscript.
