# QC pipeline

Ground-truth-free quality-control classifier pipeline for TotalSegmentator predictions:
train a model that looks at a **predicted mask alone** (no ground truth) and predicts
whether it's actually good. Six independent, resumable, CLI-driven stages (plus one
optional analysis stage) — no hardcoded paths, every parameter is a flag. Each stage is
a standalone script; a thin orchestrator (`run_pipeline.py`) runs the core six with one
consistent directory layout.

See `RESEARCH.md` (repo root, section 0 onward) for the full methodology write-up and
reasoning behind each design choice. This file is the quick developer reference.

## Stages

| # | Script | Reads | Writes | What it does |
|---|---|---|---|---|
| 1 | `run_inference.py` | dataset dir, split | `predictions/<subject>/*.nii.gz` | Runs TotalSegmentator on each subject. The only stage that touches the model. |
| 2 | `compute_metrics.py` | dataset dir + stage 1 predictions | `combined_metrics.csv`, `summary_by_organ.csv` (+ optional overlays/L-R audit/accept-analysis) | Scores predictions vs. ground truth (Dice/IoU/labels) and computes `totalsegmentator/mask_metrics.py`'s ground-truth-free features on the predicted mask. One row per (subject, organ) — this is the raw training data. CT and MR both go through this one script via `--modality {ct,mr}`. |
| 3 | `build_reference_table.py` | dataset dir (ground truth only, `train` split) | `reference_stats.csv` | Per-organ median/IQR of every feature, so "volume=50000" becomes "3 SD smaller than typical." No inference, no dependency on stage 1/2. |
| 4 | `curate_dataset.py` | one split's `combined_metrics.csv` + `reference_stats.csv` | one classifier-ready CSV (e.g. `classifier_train.csv`) | Joins organ-relative z-scores, drops empty-prediction rows, assigns the accept/reject label. Run once per split. |
| 5 | `train.py` | a curated dataset CSV | `cv_summary.csv`, `oof_predictions.csv`, `loo_by_organ.csv`, `models/*.joblib` + `training_manifest.json` | Cross-validated comparison across 4 model families × 4 feature sets + baselines, then refits every combination on the full training set and persists it to disk. |
| 6 | `test.py` | persisted models + a curated held-out test CSV | `test_results.csv`, `test_summary.csv` | Scores every persisted model against data never used in reference-building or training/CV. Reports ROC-AUC/PR-AUC plus calibration (ECE/MCE/Brier). |
| 7 *(optional)* | `ablations.py` | a curated dataset CSV from stage 4 directly | `feature_ablation.csv`, `threshold_sweep.csv`, `threshold_sweep_by_family.csv`, `threshold_sweep_family_volatility.csv`, `within_organ_auc.csv`, `within_family_auc.csv`, `family_loo.csv`, `organ_loo.csv` | Not part of the 1→6 orchestrator flow — run by hand to pressure-test whether the classifier learns real mask-quality signal or is exploiting IoU's size bias. Four experiments: feature-group ablation, IoU-threshold sweep (pooled + broken down per anatomical family, so a stable pooled AUC can't hide individual families swinging as the threshold changes), within-organ AUC, leave-one-**family**-out (stricter than stage 5's leave-one-organ-out — holds out a whole anatomical family, e.g. every rib, not just one). See `RESEARCH.md` section 6b. |

Every stage remains fully runnable on its own — `python experiments/pipeline/<stage>.py --help`.

## Running the whole thing

```
python experiments/pipeline/run_pipeline.py \
  --run-dir experiments/eval_runs/<name> \
  --dataset-dir /path/to/dataset \
  --modality {ct,mr} \
  --reference-split train --train-split <split> --test-split test
```

Computes the standard sub-paths under `--run-dir` (`predictions/`, `metrics/`,
`reference/`, `datasets/`, `models/`, `results/`) and runs stages 1–6 in order, skipping
any stage whose output already exists (`--force` to redo). Pass `--stages 4,5,6` to run
a subset. Note: the CT dataset has `train`/`val`/`test`; the MR benchmark dataset only
has `train`/`test` — see `RESEARCH.md` section 2 for how each maps onto stages 3–6.

## Resumability

Stage 1 checks per-subject whether prediction files already exist before running
TotalSegmentator — safe to kill and re-run. Stages 2–6 each write their output file
atomically at the very end of a full run, so the orchestrator's "skip if output exists"
check is always either fully-done-skip-it or not-done-run-it, never a partial state.

## Pooled, split-label-independent subject selection

`--split all` (any stage 1/2/3 script) bypasses meta.csv's split column entirely and
pools every subject in the dataset, deterministically shuffled (`common.py`'s
`POOLED_SHUFFLE_SEED`). Combine with `--offset`/`--limit` to carve out a window of that
pooled list — e.g. a QC-classifier-specific train/test split independent of whatever
train/val/test boundaries TotalSegmentator's own training used. See RESEARCH.md section
2a for why and the exact windows used.

## Shared code

`common.py` holds subject discovery (`get_subjects`/`find_columns`), canonical-RAS
image loading, calibration metrics (`ece_mce`/`brier_decomposition`), and the
`write_manifest()` helper every stage uses to record its exact CLI args + git commit
next to its output, so any file in a run directory is traceable to how it was produced.

`totalsegmentator/qc_columns.py` (not in this directory, but used throughout) defines
every column name as a shared constant — e.g. `COL_IOU = "Intersection over Union
(IoU)"` — imported by every stage so a column written by one script and read by another
is guaranteed to be the identical literal string. See `RESEARCH.md` section 6a for the
full column glossary and each column's role (feature / label-leak / identifier /
diagnostic).

## Output convention

Run outputs go under `experiments/eval_runs/<run-name>/`. Only the bulky, regenerable
binary artifacts are gitignored (`predictions/`, `models/*.joblib`, `overlays/`) —
metrics/reference/datasets/results CSVs and `*_manifest.json` files are trackable by
default, so a run's findings can be committed and shared without the large artifacts.
