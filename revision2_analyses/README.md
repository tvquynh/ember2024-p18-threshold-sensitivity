# revision2_analyses/

Second wave of analyses for the JISA Major Revision (2026-07-08 decision,
2026-09-01 deadline). Where `revision1_analyses/` re-analyzed the original
submission's stored outputs, this folder covers the parts of the revision
that required **new training runs** plus two statistical tests that were
missing from claims the original submission already made.

Everything here answers one of two questions a reviewer asked:

1. *Is the baseline actually the dataset authors' baseline?* — resolved by
   re-running the sweep under the EMBER2024 authors' released
   `lgbm_config.json`, and by reproducing their pipeline end to end.
2. *Is the mechanism anything more than "fewer training samples"?* —
   resolved by a matched-size control that removes the same number of
   malware samples at random instead of by the threshold rule.

## New training runs

`runners/run_revision2_controls.py` (repository root, `runners/`) produces
three arms, all on the same ten seeds as the main study:

| Arm | `--arm` | Thresholds | Fits | What it holds fixed |
|---|---|---|---|---|
| Canonical configuration | `canonical` | 0.065, 0.10, 0.15, 0.20, 0.30, 0.50 | 60 | Data and seeds; swaps our LightGBM configuration for the authors' released one |
| Matched-size control | `matched_size` | 0.065, 0.10, 0.12, 0.15 | 40 | Malware pool size and class prior; randomizes *which* malware are dropped |
| Authors' pipeline | `authors_pipeline` | 0.065 | 10 | Everything, including their stratified 90/10 train/validation split |

```bash
# canonical arm: the six coarse thresholds are the runner default
python runners/run_revision2_controls.py --arm canonical        --output_dir revision2_runs/rev2_canonical
python runners/run_revision2_controls.py --arm matched_size     --output_dir revision2_runs/rev2_matched
python runners/run_revision2_controls.py --arm authors_pipeline --output_dir revision2_runs/rev2_authors
```

The canonical arm covers the **coarse** grid only. The manuscript's headline
number (+4.73 pp at *T* = 0.12) sits on the LightGBM *fine* grid and is a
study-configuration result; Section 5.2 of the manuscript states this, and
Table 4 reports the canonical arm at the six coarse thresholds only. The
recommended interval *T* in [0.10, 0.15] is significant under **both**
configurations, which is the claim the canonical arm exists to support.

The runner is resume-safe (`--resume` skips completed per-run JSONs) and
writes one JSON per fit under `<output_dir>/per_run/`. Wall-clock on a
60-core CPU-only host is roughly 8.7 h for the six-threshold canonical arm,
7.0 h for the matched-size arm, and 1.7 h for the authors' pipeline arm
(17.4 h in total, 1.74 h per seed); `--prototype` runs a small subsample for
a smoke test.

All 110 resulting per-run JSONs ship in `revision2_runs/<arm>/per_run/` at the
repository root, so every analysis below re-runs without re-training.

## Analysis scripts

| File | Purpose |
|---|---|
| `analyze_revision2.py` | Seed-aligned comparison of the three control arms against the published sweep. Produces `revision2_stats.json`: paired Wilcoxon tests, Holm correction, Cohen's *d*, and the TPR/ECE blocks of paper Table 11 and Table 4. |
| `variance_pertype_tests.py` | Two tests that read only `results_aggregated/`, no retraining. (A) Pitman–Morgan paired variance-equality test for the "5–8× smaller seed-to-seed standard deviation" claim of Section 5.3. (B) Per-file-type dispersion, seed-paired Cohen's *d*, exact Wilcoxon with Holm correction inside each subtype, and per-subtype Friedman tests — paper Table 9 and its footnote. |
| `family_composition.py` | Family-level composition of the retained malware pool across the sweep: family count, unlabeled share, Herfindahl index and normalized entropy, computed both naively and over attributed samples only. Paper Table 10. |
| `pertype_variance_stats.json` | Output of `variance_pertype_tests.py`. |
| `family_composition.json` | Output of `family_composition.py`. |
| `../revision2_runs/revision2_stats.json` | Output of `analyze_revision2.py`. |

```bash
python revision2_analyses/variance_pertype_tests.py
python revision2_analyses/family_composition.py
python revision2_analyses/analyze_revision2.py \
       --results_root revision2_runs --out revision2_runs/revision2_stats.json
```

## Notes on what these analyses found

- **Fidelity is partial, and the manuscript says so.** Under the authors'
  released configuration and their 90/10 split we recover three of the
  four figures they publish for the all-PE partition within seed noise.
  The fourth, challenge PR-AUC, does not fall inside our seed spread:
  0.6404 ± 0.0020 against their 0.6354, with all ten seeds above the
  published point. Section 5.2 of the manuscript reports this as
  unreplicated rather than rounding it into agreement.

- **The discrimination effect is entirely rule-specific.** Removing the
  same number of malware samples at random is statistically
  indistinguishable from not filtering at all (all Holm-corrected
  *p* = 1.00), while threshold filtering wins on every one of thirty seed
  pairs.

- **The calibration effect is only mostly rule-specific.** Random removal
  *does* improve challenge ECE, by 5.5–9.2% against the rule-based arm's
  25.1–42.6%. Roughly a fifth of the calibration gain is a pool-size and
  class-prior effect. This finding runs against the original
  interpretation and is reported as such in Section 6.2.

- **The per-file-type zone is narrower for Win64.** All three subtypes
  peak at *T* = 0.10, but the Win64 specialist is already significantly
  worse than its own baseline at *T* = 0.15 (−1.06 pp, zero of ten seed
  pairs), where Win32 and .NET still gain. The pooled recommendation in
  Section 6.10 carries this qualification.

- **A low standard deviation on its own means little.** The
  variance-equality test also passes at *T* = 0.50 (4.5×), where the
  classifier has lost 27.80 pp of challenge TPR: consistently bad rather
  than consistently good. Dispersion is interpretable only jointly with
  the mean.
