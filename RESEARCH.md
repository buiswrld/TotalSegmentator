# RESEARCH.md

Working notes for a research paper on ground-truth-free quality control (QC) for
TotalSegmentator segmentations. Compiled from exploration/reproduction work done across
the `qc-mask-metrics`, `ct-eval-harness`, and `reproduce-ct-eval` branches. This file is a
running log of methods, rationale, and findings — not a polished paper — meant to be
referenced when writing one.

---

## 0. Pipeline architecture (staged, resumable — `experiments/pipeline/`)

The methodology in sections 3–7 below is unchanged, but as of this session the code that
implements it was restructured from five monolithic scripts (each with a hardcoded
Windows `CONFIG` block, doing several jobs inside one `main()`) into six independent,
CLI-driven, resumable stages under `experiments/pipeline/`, following the standard
ML-pipeline pattern (ingest → feature engineering → dataset curation → train → evaluate)
used by tools like TFX/Kubeflow/DVC. Every stage takes explicit `--dataset-dir`/
`--output-*` flags — nothing is hardcoded — and can be re-run independently, picking up
from any prior stage's output on disk. A thin orchestrator
(`experiments/pipeline/run_pipeline.py --run-dir <dir> --dataset-dir <dir> --modality
{ct,mr}`) runs all six in sequence with a consistent directory layout, skipping any
stage whose output already exists (pass `--force` to redo it); every stage remains
independently runnable with fully custom paths too.

| Stage | Script | Absorbed from (now retired) |
|---|---|---|
| 1. Inference | `run_inference.py` | `ensure_predictions()` in `combined.py`/`evaluate_ct.py` |
| 2. Compute metrics | `compute_metrics.py` | `evaluate_ct.py`/`combined.py` (CT+MR merged into one modality-aware script) |
| 3. Build reference table | `build_reference_table.py` | `experiments/build_reference.py` |
| 4. Curate dataset | `curate_dataset.py` *(new)* | `train_classifier.py::load()` |
| 5. Train | `train.py` | `train_classifier.py`, **+ new: persists fitted models to disk** |
| 6. Test | `test.py` *(new)* | `calibrate.py`'s ECE/MCE/Brier functions (now in `common.py`), applied to a real held-out test set rather than nested CV |

The `EXPECTED_DICE` benchmark table (previously hardcoded in `combined.py`) now lives at
`resources/expected_dice_mr.json`, loaded via `compute_metrics.py --expected-dice-json`.

References to `combined.py`/`evaluate_ct.py`/`build_reference.py`/`train_classifier.py`/
`calibrate.py` elsewhere in this document (sections 3–10) describe the methodology as
originally implemented and are kept as an accurate historical record of what actually
produced those specific findings — they are not run instructions. For a current run, use
the `experiments/pipeline/` stage scripts above.

---

## 1. The overall research question

TotalSegmentator produces segmentations with no ground truth available at deployment
time (a real clinical/research scan has no expert-drawn mask to compare against). The
question this line of work is chasing:

> Can we build a classifier that looks at a **predicted mask alone** (no ground truth)
> and reliably flags "this segmentation is probably wrong," well enough that its
> confidence score can be trusted by a non-specialist (e.g. a clinician)?

This requires, in order:
1. A way to measure a mask's own properties without needing ground truth (feature
   extraction).
2. A way to generate *labeled* training data — i.e., actually compare predictions to
   ground truth, but only to build the training set, never at inference time.
3. A way to give those raw features organ-specific context (a "big" pancreas and a "big"
   liver mean very different things).
4. A model that learns to predict quality from features alone.
5. A way to verify the model's confidence scores are honest, not just well-ranked.

Each stage below corresponds to one file in `experiments/` / `totalsegmentator/`.

---

## 2. Datasets

Two public TotalSegmentator benchmark datasets, downloaded locally to an external drive
(`/Volumes/Datasets/`), each with `meta.csv` (subject id + train/val/test split) and one
subfolder per subject containing the image + a `segmentations/` folder of one `.nii.gz`
per ground-truth organ.

### MR dataset — `TotalsegmentatorMRI_dataset_v200`
- 616 subjects total: **561 train / 55 test**. No `val` split.
- Per subject: `mri.nii.gz` + `segmentations/*.nii.gz` (~50 organs).
- This is the dataset behind Akinci D'Antonoli et al., *Radiology* 2025
  (doi:10.1148/radiol.241613, arXiv:2405.19492) — the `total_mr` model's own paper.
- The 55-subject `test` split matches the paper's own "55-subject internal test set"
  description, confirmed by subject count — this is very likely the *exact* held-out set
  the paper's published per-organ Dice numbers (`EXPECTED_DICE` in `combined.py`) were
  measured on, which is what makes a direct reproduction comparison meaningful.

### CT dataset — `Totalsegmentator_dataset_v201`
- 1,082 train / 89 test / **57 val**. Has all three splits, unlike the MR dataset.
- Per subject: `ct.nii.gz` + `segmentations/*.nii.gz`.
- Used by the `ct-eval-harness` pipeline: `train` split builds the organ reference table
  (Section 4), `val` split trains/evaluates the classifier, `test` split is reserved
  untouched for a final evaluation.

**Open question for the paper:** the MR dataset has no `val` split, so the CT pipeline's
clean 3-way separation (train → reference table, val → classifier, test → final eval)
can't be replicated as-is on MR data. Will need to either carve a held-out subset out of
MR `train`, or restructure how MR fits into the same methodology.

---

## 3. Stage 1 — Ground-truth-free feature extraction (`totalsegmentator/mask_metrics.py`)

Given one predicted mask (and, for intensity features, the source CT/MR volume), computes
descriptive features that require **no ground truth**:

- **Volume metrics** — voxel count, physical volume (mm³), whether the mask is empty.
- **Shape metrics** — centroid position (relative to image bounds), bounding-box
  dimensions (relative), bounding-box volume, mask-to-bbox fill ratio.
- **Component metrics** — number of connected components, fraction of the mask taken up
  by the largest component (a highly fragmented mask is a red flag).
- **Boundary metrics** — whether the mask touches the image boundary (possible
  truncation/FOV issue), boundary roughness fraction.
- **Intensity metrics** — mean/median/std/p05/p95 HU (or MR intensity) values under the
  mask — an organ mask with implausible intensity statistics is suspect.

These are the columns later called "raw"/"geom-only" features in the classifier scripts.
Everything downstream is built on top of this module.

---

## 4. Stage 2 — Labeled evaluation harnesses

Two parallel scripts, MR and CT, same core design:

### `combined.py` (MR) — `sugoiiia-patch-1` → `qc-mask-metrics`
Per subject: ensure predictions exist (run TotalSegmentator if not), compute per-organ
**Dice** and **IoU** against ground truth, classify each organ-case as `ok` / `low_dice`
(Dice < 0.50) / `miss` (GT has it, prediction doesn't) / `false_positive` (predicted,
but GT doesn't have it), save a 3-plane overlay PNG.

**Built-in correctness check:** every scored row is verified against the algebraic
identity `IoU = Dice / (2 - Dice)`; the run hard-aborts (`SystemExit`) if this doesn't
hold — this guards against a broken TP/FP/FN counting bug silently producing wrong
numbers, rather than a subtle statistical check.

**`EXPECTED_DICE` benchmark table** ([combined.py:43-63]) — hardcoded per-organ Dice
values, copied from `resources/results_all_classes_mr.json` in the upstream
`wasserth/TotalSegmentator` repo (the `total_mr` model's own published benchmark, from
the Radiology 2025 paper, measured on their 55-subject internal test set). Important
caveat for the paper: **this is a population-level average from a different test set**,
not a per-scan ground-truth expectation — there is no mechanism anywhere in this codebase
for "expected Dice for this specific scan." A delta between our measured Dice and this
table could reflect either a real pipeline problem or genuine difficulty differences
between datasets, which is why...

**`diagnose_pipeline()`** exists as an automatic gate: if >30% of matched organs fall
>0.15 Dice below `EXPECTED_DICE` ("systematic shortfall"), the script automatically
inspects affine alignment (pred vs. GT vs. source image), organ-name overlap, and
shape/spacing — to rule out a pipeline bug (misalignment, wrong orientation, wrong task)
before concluding the model is genuinely underperforming.

### `experiments/evaluate_ct.py` (CT) — `ct-eval-harness`
Same skeleton, adapted for CT (`total` task, `ct.nii.gz`, HU windowing for overlay
display), with several additions beyond the MR script:

- **Canonical RAS reorientation** (`nib.as_closest_canonical`) applied to every volume
  before any voxel comparison, so Dice/IoU and positional features are computed in one
  consistent anatomical frame regardless of how a given scan was originally stored.
- **Feature computation is fused into the same pass** — for every scored organ-mask, it
  also computes the Stage-1 `mask_metrics` features on the **predicted** mask (not GT).
  This produces `combined_metrics.csv`: one row per subject-organ with *both* the
  accuracy label (dice/iou/accept) and the ground-truth-free features — i.e., this file
  *is* the classifier's training data.
- **Left/right anatomical audit** (`lr_rows()`) — for every paired `*_left`/`*_right`
  structure, checks that in canonical RAS (+x = patient right), the `_right` structure's
  centroid actually has a higher x-coordinate than `_left`'s. Flags `SWAPPED` if not, run
  on both predictions and ground truth. Overlay images also stamp "pt LEFT"/"pt RIGHT"
  labels on the axial view so a labeling bug is visible at a glance.
- **Accept/reject labeling** — introduces `IOU_ACCEPT = 0.90`: any organ-mask with
  IoU ≥ 0.90 is labeled "accept," otherwise "reject." This is the binary label the
  classifier is eventually trained to predict. `accept_analysis.csv` breaks down accept
  rate and mean IoU per organ, plus IoU percentiles.
- No `EXPECTED_DICE`-style benchmark comparison (unlike the MR script) — this script's
  job is purely to generate labeled training data, not validate against a published
  number.

---

## 5. Stage 3 — Organ-specific reference table (`experiments/build_reference.py`)

**Problem it solves:** a raw feature like `volume_mm3 = 50000` is meaningless without
organ context — normal for a liver, catastrophic for a gallbladder.

**How it works:** reads **ground-truth masks only** (no segmentation model, no GPU
needed) from the CT dataset's **`train`** split (200 of 1,082 subjects by default).
For every organ, computes the Stage-1 features on the *true* mask across all sampled
subjects, then collapses each (organ, feature) pair into robust summary statistics:
median, Q25, Q75, IQR. Volume-like features (`num_voxels`, `volume_mm3`,
`bbox_volume_mm3`) are log-transformed (`log1p`) first since they're right-skewed, so the
computed spread isn't dragged around by a few outlier subjects (this is recorded per row
via the `log_space` flag).

**Output — `reference_stats.csv` (long format), one row per organ×feature:**
| column | meaning |
|---|---|
| `organ` | which anatomical structure |
| `feature` | which measurement (e.g. `volume_mm3`, `mean_HU`) |
| `n` | number of ground-truth observations behind this row |
| `median` | typical value for this organ-feature — the reference center point |
| `q25` / `q75` | interquartile bounds |
| `iqr` | `q75 - q25`, spread measure — used instead of std dev because it's robust to outliers |
| `log_space` | whether values were log1p-transformed before computing stats |

**Applying it — `add_relative_features()`:** converts a new mask's raw feature into a
robust z-score: `z = (x - median) / (iqr / 1.349)`. The 1.349 constant converts an IQR
into a standard-deviation equivalent for a normal distribution, so `z = 3` reads on the
familiar "3 SDs off typical" scale while staying robust to outliers. Raw columns are kept
alongside the new `_z` columns, not replaced (tree models handle the redundancy fine, and
some raw values carry information the z-score discards). Organs missing from the
reference table (too few `train`-split observations) get `NaN` z-scores, handled
downstream by median imputation.

**Excluded from z-scoring, on purpose:** `num_components`, `largest_component_fraction`,
`touches_boundary`, `boundary_fraction` — these already mean the same thing for any
organ (e.g. "1 connected component" is universally good), so no reference/normalization
is needed; kept raw in the classifier's organ-relative feature set (`ORGAN_FREE_RAW`).

**Critical design decision, worth emphasizing in the paper:** the reference table is
built from `train` only, then **frozen** — treated exactly like a training-set
normalization constant (analogous to computing a mean/std for standardization). This
keeps the ground-truth-free claim intact at deployment: scoring a new scan only requires
*looking up* this precomputed table, never touching that scan's own ground truth (which
doesn't exist anyway in real deployment). It also prevents the reference stats from
leaking information into whatever split trains/evaluates the classifier.

---

## 6. Stage 4 — Classifier training (`experiments/train_classifier.py`)

**Goal:** predict the accept/reject label (Stage 2) from ground-truth-free features
(Stage 1 raw + Stage 3 organ-relative), and rigorously check whether the model is
learning real geometric signal or just memorizing "which organs are usually easy."

### Dataset construction
Reads `combined_metrics.csv` (Stage 2 output). Columns split into:
- **Leak columns** (excluded from features, `dice/iou/tp/fp/fn/gt_vox/status/accept`) —
  anything computed from ground truth. `iou >= IOU_ACCEPT` becomes the `label` column.
  A hard `assert` checks no leak column ends up in the feature set.
- **ID columns** (`subject/organ/orig_axcodes/pred_vox`) — bookkeeping, not features
  (though `organ` gets reused for one-hot encoding in one feature-set variant).
- **Raw features** — everything else = the Stage-1 `mask_metrics` columns.
- **Organ-relative features** — the `_z` columns added by joining Stage-3's reference
  table via `add_relative_features()`.

**Preprocessing before training:**
- Rows with `is_empty == 1` (no predicted mask at all) dropped by default — reasoning:
  "nothing predicted" is trivially handled by a simple rule, not worth a classifier.
- **Median imputation** on every numeric feature — required because `_z` columns are
  `NaN` for organs under-represented (or absent) in the reference table.
- **Standard scaling** — applied only for logistic regression (distance/gradient-based,
  sensitive to feature magnitude mismatches — e.g. `volume_mm3` in the tens of thousands
  vs. `mask_to_bbox_ratio` in [0,1]). Tree-based models are scale-invariant by
  construction (they split on thresholds, not distances), so they skip this.
- **Organ one-hot encoding** — only in the `raw+organ` feature-set variant. Expands the
  categorical `organ` column into one binary column per organ, avoiding a false ordinal
  relationship a plain integer encoding would imply.
- **Class-imbalance weighting** (`class_weight="balanced"` / `"balanced_subsample"` for
  random forest) — reweights each class inversely proportional to its frequency in
  training, so the rarer "reject" class isn't drowned out by a model that could otherwise
  get high accuracy by just always predicting "accept."

### Feature-set ablation (the actual experiment)
Four (or two, in the smaller pilot run — see Section 7) named column subsets, each
trained separately across multiple model families (logistic regression,
depth-3 decision tree, random forest, hist-gradient-boosting):

| name | columns | organ one-hot? |
|---|---|---|
| `raw` / `geom-only` | Stage-1 raw features only | no |
| `raw+organ` / `+organ` | same + organ identity | yes |
| `rel` | organ-relative z-scores + organ-free raw features | no |
| `raw+rel` | raw + z-scores combined | no |

Comparing across these subsets — not just picking one "best" model — is the actual
research question: does organ identity alone explain most of the accuracy (`raw` vs.
`raw+organ` gap), and do organ-relative features carry real signal that generalizes
(`rel`'s standalone performance, tested further by leave-one-organ-out).

### Grouped cross-validation
`GroupKFold` on `subject` (not plain random K-fold). Rows from the same subject (many
organs per subject) always land entirely in one fold — never split across train/test.
**Why this matters:** rows from the same patient share scanner settings, body habitus,
noise characteristics; a random split could let the model partially recognize "this is
subject X's scan" rather than learning general mask-quality signal, inflating the
apparent test score in a way that wouldn't hold on a genuinely new patient.

### Baselines — the diagnostic core of the experiment
- **`always-accept`** — predicts the constant training-fold accept rate for every row,
  regardless of features. The absolute floor; AUC is defined as exactly 0.5 (a constant
  score has no ranking power).
- **`organ-rate baseline`** — predicts each organ's *historical* accept-rate (learned
  from training folds only), with **zero mask geometry** — it never looks at the actual
  mask, only which organ it is. Because it assigns the *identical* score to every mask of
  a given organ, it structurally has **zero ability to rank within an organ** (can't
  distinguish a good pancreas mask from a bad one). This makes it a clean, provable test:
  any real improvement a geometry-aware model shows over this baseline can only come from
  correctly discriminating within-organ, which the baseline literally cannot do. A real
  model that fails to clear this baseline has learned nothing but an organ lookup table
  — a critical distinction, since organ difficulty alone (e.g. pancreas being inherently
  harder than liver) could otherwise make a "cheating" model look deceptively skilled.

**Generalizing this idea for the paper:** the organ-rate baseline is a special case of a
broader principle — checking whether a strong categorical confound explains apparent
model skill. It got special treatment here because organ identity is the cleanest,
strongest possible confound (small number of discrete classes, `groupby().mean()` gives
a perfect shortcut) and is the specific axis the whole project's generalization claim
rests on. `importances()` (feature-importance ranking) serves an analogous, more graded
diagnostic role across *all* features — e.g. it would be worth explicitly testing whether
`largest_component_fraction` (flagged in code comments as "the single strongest feature")
dominates the same way organ risked doing, via a similar single-feature ablation.

### Leave-one-organ-out (LOO)
Trains on every organ *except* one, tests on the held-out organ — using only
organ-agnostic feature sets (no one-hot organ column, since the point is testing transfer
to an organ the model has literally never seen a training example of). The hypothesis:
raw features can't transfer (the model has no idea what scale is normal for an unseen
structure), but organ-relative z-scores should, because "3 SD below typical" means the
same thing regardless of which organ it's attached to. Organs need a minimum row count
(`LOO_MIN_N=25`) and minimum class balance (`LOO_MIN_MINORITY=5`) to be included, since an
AUC computed on too few rows is unreliably noisy.

### Reported outputs
- **`final_training_dataset.csv`** — written before any model fitting, from the exact
  in-memory DataFrame (`ID_COLS + raw features + organ-relative z-scores + training
  label`) that gets handed to cross-validation. Exists purely so a developer can open
  one clean, fully-labeled CSV and manually review precisely what the classifier is
  being trained on, rather than reconstructing it from scattered function calls.
- Ranked table: ROC-AUC, PR-AUC(accept), PR-AUC(reject) per model, with fold std-dev.
  **PR-AUC(reject) is the more important column given the class imbalance** — accept-side
  metrics are inflated by the trivially large majority class.
- Operating-point table for the best model: at several probability thresholds, the
  precision/recall/flagged-for-review tradeoff — the practical deployment question of how
  much manual review workload to accept in exchange for auto-accept confidence.
- Out-of-fold (OOF) predictions CSV — every row's prediction comes from the one CV fold
  where it was held out, giving one honest, never-trained-on prediction per row; this
  feeds `calibrate.py`.
- Feature importance ranking (whole-dataset random forest fit, diagnostic only, not
  cross-validated).

---

## 6a. Feature & Metric Glossary

Every column that appears anywhere in this pipeline's CSVs, defined once as a constant in
`totalsegmentator/qc_columns.py` (see that file's docstring for how it's used — it's a
shared source of truth for the literal column-name strings, not a translation layer).

**This table must be kept up to date.** Whenever a new feature/metric column is added
anywhere in this pipeline: add its constant to `totalsegmentator/qc_columns.py`, then add
its row here with the correct role (see `CLAUDE.md` for the standing instruction to do
this automatically).

**Role** — `feature`: fed to the classifier as an input. `label/leak`: ground-truth-derived,
used only to build the accept/reject label, excluded from training features (see
`LEAK_COLS` in `train.py`/`test.py`). `identifier`: bookkeeping, not
predictive. `diagnostic`: computed for the experiment/paper (L/R audit, reference-table
build stats, calibration metrics) but never fed to the classifier.

| Constant | Long label (short form) | Origin | Role |
|---|---|---|---|
| `COL_SUBJECT` | Subject ID (subject) | all stages | identifier |
| `COL_ORGAN` | Organ/Structure Name (organ) | all stages | identifier |
| `COL_ORIG_AXCODES` | Original Image Axis Orientation Codes (orig_axcodes) | `compute_metrics.py` | identifier |
| `COL_NUM_VOXELS` | Number of Voxels (num_voxels) | `mask_metrics.py` | feature |
| `COL_VOLUME_MM3` | Volume in Cubic Millimeters (volume_mm3) | `mask_metrics.py` | feature |
| `COL_IS_EMPTY` | Is Mask Empty (is_empty) | `mask_metrics.py` | feature (also used to drop empty-prediction rows before training) |
| `COL_CENTROID_X_REL` / `Y_REL` / `Z_REL` | Relative Centroid X/Y/Z Position | `mask_metrics.py` | feature |
| `COL_BBOX_X_REL` / `Y_REL` / `Z_REL` | Relative Bounding Box Width/Height/Depth | `mask_metrics.py` | feature |
| `COL_BBOX_VOLUME_MM3` | Bounding Box Volume in Cubic Millimeters (bbox_volume_mm3) | `mask_metrics.py` | feature |
| `COL_MASK_TO_BBOX_RATIO` | Mask-to-Bounding-Box Fill Ratio (mask_to_bbox_ratio) | `mask_metrics.py` | feature |
| `COL_NUM_COMPONENTS` | Number of Connected Components (num_components) | `mask_metrics.py` | feature (organ-free, no reference normalization) |
| `COL_LARGEST_COMPONENT_FRACTION` | Largest Connected Component Fraction (largest_component_fraction) | `mask_metrics.py` | feature (organ-free; called out in code as the single strongest feature) |
| `COL_TOUCHES_BOUNDARY` | Touches Image Boundary (touches_boundary) | `mask_metrics.py` | feature (organ-free) |
| `COL_BOUNDARY_FRACTION` | Boundary Voxel Fraction (boundary_fraction) | `mask_metrics.py` | feature (organ-free) |
| `COL_MEAN_HU` / `MEDIAN_HU` / `STD_HU` / `P05_HU` / `P95_HU` | Mean/Median/Std/5th/95th Percentile Hounsfield Unit Intensity | `mask_metrics.py` | feature |
| `COL_DICE` | Sorensen-Dice Coefficient (Dice) | `compute_metrics.py` | label/leak |
| `COL_IOU` | Intersection over Union (IoU) | `compute_metrics.py` | label/leak (thresholded to build `COL_TRAINING_LABEL`) |
| `COL_TRUE_POSITIVES` / `FALSE_POSITIVES` / `FALSE_NEGATIVES` | True/False Positive/Negative Voxel Count (TP/FP/FN) | `compute_metrics.py` | label/leak |
| `COL_PREDICTED_VOXEL_COUNT` | Predicted Mask Voxel Count (pred_vox) | `compute_metrics.py` | identifier (not ground-truth-derived, but kept in `ID_COLS` rather than the feature set) |
| `COL_GROUND_TRUTH_VOXEL_COUNT` | Ground Truth Mask Voxel Count (gt_vox) | `compute_metrics.py` | label/leak |
| `COL_MATCH_STATUS` | Match Status (status) | `compute_metrics.py` | label/leak |
| `COL_ACCEPT_LABEL` | Accept/Reject Label (accept) | `compute_metrics.py` | label/leak |
| `COL_TRAINING_LABEL` | Accept/Reject Training Label (label) | `train.py`/`test.py` | this IS the training target, derived from `COL_IOU` |
| `COL_FEATURE_NAME` | Feature/Metric Name (feature) | `build_reference_table.py` | diagnostic (reference table only) |
| `COL_OBSERVATION_COUNT` | Number of Reference Observations (n) | `build_reference_table.py` | diagnostic |
| `COL_REFERENCE_MEDIAN` / `Q25` / `Q75` / `IQR` | Reference Median/25th/75th Percentile/Interquartile Range | `build_reference_table.py` | diagnostic (used to compute `_z` features, not fed to the model directly) |
| `COL_LOG_SPACE_FLAG` | Computed in Log Space (log_space) | `build_reference_table.py` | diagnostic |
| `<feature>_z` (e.g. `COL_VOLUME_MM3 + "_z"`) | organ-relative z-score of the given feature | `build_reference_table.py::add_relative_features` | feature |
| `COL_MASK_SOURCE` | Mask Source: Prediction or Ground Truth (source) | `compute_metrics.py` L/R audit | diagnostic |
| `COL_PAIRED_STRUCTURE` | Paired Structure Base Name (structure) | `compute_metrics.py` L/R audit | diagnostic |
| `COL_LEFT_CENTROID_X_REL` / `COL_RIGHT_CENTROID_X_REL` | Left/Right Structure Relative Centroid X | `compute_metrics.py` L/R audit | diagnostic |
| `COL_RIGHT_MINUS_LEFT` | Right Minus Left Centroid X Difference | `compute_metrics.py` L/R audit | diagnostic |
| `COL_LR_VERDICT` | Left/Right Orientation Verdict (verdict) | `compute_metrics.py` L/R audit | diagnostic |
| `COL_N_PRESENT_BOTH` / `COL_N_SCORED` | count present in both pred+GT / count scored | summary tables | diagnostic (rollup, not per-row feature) |
| `COL_MEAN_DICE` / `COL_MEAN_IOU` | Mean Dice / Mean IoU | summary tables | diagnostic |
| `COL_N_OK` / `COL_N_LOW_DICE` / `COL_N_MISS` / `COL_N_FALSE_POSITIVE` | status-breakdown counts | summary tables | diagnostic |
| `COL_N_ACCEPT` / `COL_ACCEPT_RATE` | accept count / accept rate | summary + accept_analysis + LOO tables | diagnostic |
| `COL_MEASURED_DICE` / `COL_EXPECTED_DICE` / `COL_DICE_DELTA` / `COL_N_SUBJECTS` | measured vs. published `EXPECTED_DICE` benchmark comparison | `compute_metrics.py` (MR only) | diagnostic |
| `COL_N_MASKS` | Number of Masks (n) | `accept_analysis.csv`, `test.py` bins | diagnostic |
| `COL_MEAN_IOU_ACCEPTED` / `COL_MEAN_IOU_REJECTED` | mean IoU within the accepted/rejected group | `accept_analysis.csv` | diagnostic |
| `COL_MODEL_NAME` / `COL_CALIBRATION_METHOD` | which base model / which calibration method | `test.py` | diagnostic |
| `COL_AUC` / `COL_ECE` / `COL_MCE` / `COL_BRIER_SCORE` | ranking + calibration-error metrics | `test.py` | diagnostic |
| `COL_RELIABILITY` / `COL_RESOLUTION` / `COL_UNCERTAINTY` | Brier score decomposition components | `test.py` | diagnostic |
| `COL_BIN_MEAN_PREDICTED` / `COL_BIN_OBSERVED_RATE` / `COL_CALIBRATION_GAP` | per-bin calibration-curve values | `test.py` | diagnostic |

Not in this table: `EXPECTED_DICE`'s keys in `compute_metrics.py` (organ names, not metric
short-forms) and every dynamically-named per-model/per-feature-set result column
(`oof_predictions.csv`, `loo_by_organ.csv`, `calibrated_oof.csv` — named
`<model family> [<feature set>]` or `<base model> + <calibration method>`, can't be
static constants since the set of models/feature-sets is configured at runtime).

## 7. Stage 5 — Calibration (`experiments/calibrate.py`)

**Motivation (from the script's own docstring):** AUC measures whether the model
*ranks* masks correctly; it says nothing about whether a score of 0.8 really means "80%
chance this mask is good." For a tool meant to communicate reliability to a
non-specialist (e.g. "85% chance this mask is within tolerance" — a sentence a clinician
could act on), calibration matters more than ranking quality alone.

Computes ECE (Expected Calibration Error) / MCE (Maximum Calibration Error) / Brier score
decomposition, produces a reliability diagram (`calibration.png`) and out-of-fold
calibrated probabilities (`calibrated_oof.csv`), comparing calibration methods (none /
sigmoid / isotonic) via nested out-of-fold evaluation. **Not yet deeply explored in this
session — worth a dedicated pass once we have our own reproduced classifier results to
calibrate.**

---

## 8. Findings so far (reproduced ourselves this session)

### MR reproduction run — `combined.py`, full 55-subject `test` split, `total_mr` model
Ran on the downloaded MR dataset (`/Volumes/Datasets/TotalsegmentatorMRI_dataset_v200`),
full-resolution model (not the fast 3mm variant, deliberately — needed for a fair
comparison against `EXPECTED_DICE`, which was itself measured with the full model), MPS
device (Apple GPU) for speed. Branch: `qc-mask-metrics` (script pulled onto
`ct-eval-harness` for the run).

- **Correctness check passed:** IoU/Dice algebraic identity held for all scored rows,
  Spearman r = 1.000000 between Dice and IoU.
- **Benchmark comparison:** 50/50 organs matched against `EXPECTED_DICE`. Only **2
  organs (4%)** fell >0.15 Dice below the published benchmark — well under the 30%
  "systematic shortfall" threshold, so the automatic `diagnose_pipeline()` never
  triggered. **Conclusion: our local pipeline reproduces the published MR benchmark
  closely for the paper's own held-out test set.**
  - `portal_vein_and_splenic_vein`: measured 0.606 vs. expected 0.766 (Δ −0.160)
  - `inferior_vena_cava`: measured 0.743 vs. expected 0.901 (Δ −0.159)
  - Both are thin, low-volume vascular structures — Dice is naturally more volatile for
    small structures (a few boundary voxels swing the score more), so this isn't
    necessarily alarming on its own; worth a note in the paper about metric volatility
    scaling inversely with structure size.
- **Worst organs overall (not just vs. benchmark):** pancreas (0.614),
  portal_vein_and_splenic_vein (0.606), iliac_vena_left (0.658), clavicula_right (0.660),
  adrenal_gland_left (0.686) — consistently the smallest/thinnest structures.
- **Six subjects returned 0 scoreable organs** (`s0078, s0458, s0480, s0523, s0580,
  s0150`) — nothing passed the `MIN_VOXELS`/shape-match filter. **Not yet root-caused —
  open item.** Worth checking whether this is a real shape-mismatch issue or a benign
  small-organ/FOV effect before writing this up.
- **Operational note:** the first full attempt at this run was killed silently when the
  backing session/process was torn down mid-run (not a script bug — no error/traceback,
  just an abrupt stop). Confirmed via `ensure_predictions()`'s caching behavior that a
  resumed run correctly skipped the 47 already-completed subjects and only recomputed the
  remaining ~8, then proceeded to scoring. Relaunching detached (`nohup ... & disown`)
  fixed the reliability of long unattended runs going forward.

### CT classifier pilot run (colleague's run, `train_classifier.py`, ~1,459 masks / 20 subjects)
Small pilot (20 subjects — likely early/dev-scale, worth re-running at full `val`-split
scale of 57 subjects for the paper). Only two feature-set variants present
(`geom-only`, `+organ`) — no `rel`/`raw+rel` variants, meaning `reference_stats.csv`
either wasn't present for this run or this used an earlier script variant. **Open item:
confirm with the collaborator, and re-run with the reference table joined for a full
4-feature-set comparison.**

Results:
| model | ROC-AUC | PR-AUC(acc) | PR-AUC(rej) |
|---|---|---|---|
| randomforest [+organ] | 0.843 | 0.941 | 0.611 |
| randomforest [geom-only] | 0.835 | 0.937 | 0.590 |
| logreg [+organ] | 0.831 | 0.936 | 0.577 |
| logreg [geom-only] | 0.813 | 0.935 | 0.514 |
| histgradboost (both variants, identical) | 0.807 | 0.916 | 0.559 |
| tree-depth3 (both variants, identical) | 0.794 | 0.913 | 0.494 |
| **organ-rate baseline** | 0.773 | 0.906 | 0.474 |
| always-accept | 0.500 | 0.800 | 0.167 |

- **Every real model beat the organ-rate baseline**, including the weakest
  `geom-only` variant with no organ column at all — meaning even without any way to
  memorize organ identity, models extract real predictive signal from mask geometry.
  This is the key positive finding for the "not just cheating" claim.
- **The `[+organ]` vs `[geom-only]` gap is small** (≤0.02 AUC across the board) —
  organ identity is a minor boost on top of geometry, not the dominant driver. Reinforces
  that geometric features are doing most of the real work.
- Best model: `randomforest [+organ]`, ROC-AUC 0.843, PR-AUC(reject) 0.611 vs. baseline's
  0.474 — a meaningful improvement specifically at catching bad masks (the actual task).
- **Operating points** (best model): threshold 0.30 → 98.7% recall / 84.2% precision /
  6.1% flagged for review; threshold 0.80 → 64.5% recall / 93.5% precision / 44.8%
  flagged. Classic precision/recall/review-workload tradeoff — useful for the paper's
  discussion of deployment thresholds.
- **Feature importance:** `centroid_y_rel` ranked highest (0.1033) — positional deviation
  is the single strongest predictor of mask quality, ahead of any size feature. Followed
  by a cluster of size-related features (`bbox_volume_mm3`, `bbox_y_rel`, `volume_mm3`,
  `num_voxels`). **Discrepancy to flag:** `largest_component_fraction` — called "the
  single strongest feature" in the script's own code comment — only ranked 7th (0.0586)
  in this run. Possibly comment reflects a different/larger run; worth re-checking at
  full scale.

---

## 9. Open questions / ideas to pursue for the paper

- **MR dataset has no `val` split** — need a decision on how to adapt the
  train→reference / val→classifier / test→final-eval methodology to MR data, or scope
  the paper's classifier experiments to CT only and use MR purely for the
  benchmark-reproduction result (Section 8).
- **Root-cause the 6 zero-organ MR subjects** before citing the MR reproduction numbers
  in the paper — confirm whether they're a real data issue or expected.
- **Re-run the CT classifier pilot at full scale** (57-subject `val` split, not 20) with
  the reference table joined (`rel`/`raw+rel` feature sets included) for the paper's main
  classifier results — the current pilot is too small and missing half the planned
  feature-set ablation.
- **Extend the "beat the baseline" diagnostic beyond organ identity** — consider a
  single-feature ablation/baseline for `largest_component_fraction` (and possibly
  `centroid_y_rel`, given its unexpectedly high importance in the pilot) to check whether
  the model is over-relying on one dominant feature the same way it could over-rely on
  organ identity.
- **Run `calibrate.py`** once a full-scale classifier result exists — not yet explored in
  this session; calibration (not just ranking quality) is central to the paper's framing
  around communicating reliability to non-specialists.
- **Two divergent versions of the MR harness exist** — `master` has an original
  253-line version (`8c01e33`, committed directly by Aahil), `qc-mask-metrics` has a
  later refactored ~440-line version (from `sugoiiia-patch-1`). Should reconcile which is
  canonical before the paper cites either's methodology.
- **No CSV/dataset artifacts are version-controlled anywhere in the repo** — every run
  output (`combined_metrics.csv`, `reference_stats.csv`, `oof_predictions.csv`,
  `calibration_metrics.csv`, etc.) is a local artifact on whoever ran it. For
  reproducibility in the paper, we should decide on a shared location/process for
  archiving the specific run outputs the paper's numbers are drawn from.

---

## 10. Repo / branch map (as of this session)

- `master` — upstream TotalSegmentator + the original (unrefactored) MR harness commit.
- `qc-mask-metrics` — `mask_metrics.py` QC module, RAS reorientation, `calculate_mask_volumes`,
  refactored `combined.py` (MR harness).
- `ct-eval-harness` — built on `qc-mask-metrics`; adds `evaluate_ct.py` (CT harness),
  `build_reference.py`, `train_classifier.py`, `calibrate.py` — the full pipeline this
  document describes.
- `reproduce-ct-eval` — branched from `ct-eval-harness`, created this session as the
  working branch for reproducing/validating these results ourselves.
- `restore-local-config-notes` — small branch restoring local `.gitignore`/`CLAUDE.md`
  notes about the `experiments/eval_runs/` output convention (not yet merged into
  `reproduce-ct-eval`).
- `sugoiiia-patch-1`, `test-tool` — earlier/superseded branches, mostly subsumed into
  `qc-mask-metrics`.
- `origin/add-gitignore` — small unmerged remote branch, `.gitignore` additions for
  NIfTI/predictions/outputs.
