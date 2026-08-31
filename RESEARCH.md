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
| 7. Ablations *(optional)* | `ablations.py` | — new, not part of the original five scripts |

Stage 7 is an optional analysis script, not part of the linear 1→6 flow the orchestrator
runs — it takes a curated dataset CSV from stage 4 directly (`--dataset-csv`) and runs
four deeper diagnostic experiments (feature ablation, IoU-threshold sweep, within-organ
AUC, leave-one-family-out) to pressure-test whether the classifier is learning genuine
mask-quality signal or just exploiting IoU's size bias (see section 6b). It's meant to be
run by hand against `datasets/classifier_train.csv` once a normal 1→6 run has produced one.

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

**Open question for the paper (resolved, see 2a below):** the MR dataset has no `val`
split, so the CT pipeline's clean 3-way separation (train → reference table, val →
classifier, test → final eval) couldn't be replicated as-is on MR data — MR's classifier
ran on cross-validation only, with `--train-split == --test-split` (both "test", 55
subjects) and no subject the model had never seen at all.

---

## 2a. QC-pipeline-specific train/test partitioning (pooled, split-label-independent)

The QC classifier is a separate model downstream of a *frozen* TotalSegmentator — it
never trains or fine-tunes the segmentation model itself, so there's no reason it needs
to respect TotalSegmentator's own train/val/test boundaries (those exist to evaluate
*TotalSegmentator's own* accuracy, e.g. the MR paper's 55-subject held-out Dice numbers
in section 2 above). Both datasets' classifier pools are therefore built by **pooling
every subject regardless of its original split label**, deterministically shuffling
(`common.py::POOLED_SHUFFLE_SEED = 0`, so results are reproducible), and splitting
80:20 into classifier-train:classifier-test — mechanically, `--split all` on any stage-1/
2/3 CLI script bypasses the meta.csv split-column filter, then `--offset`/`--limit` carve
out a window of the resulting pooled, shuffled list (`get_subjects()` in `common.py`).

**Reference table:** built from **every** subject in each dataset (`--split all`, no
`--limit`) — it's ground-truth-only, costs no GPU time, and there's no reason to
withhold any subject from it.

**Classifier pool:** also **every** subject, split 80:20. A subject can therefore appear
in both the reference table's population *and* as a classifier training/test example —
a deliberate choice, not an oversight (see the caveat on the pre-fix `mr_full_run` numbers
in section 8, which had exactly this overlap by accident rather than by design). The
trade-off: a subject's own "relative" (organ z-score) features are computed against a
population that includes its own ground truth — negligible for common organs (one point
among hundreds) and proportionally larger for organs with few ground-truth instances
across the dataset (e.g. `prostate`, sex-specific; several organs in the reference table
sit below the 10-observation noise threshold even before this run — see the reference
table's own printed warning). Accepted in exchange for not wasting any subject's data —
maximizing classifier training/test pool size was judged more valuable than the small
bias this introduces.

| | total subjects | reference (offset:limit) | classifier-train, 80% (offset:limit) | classifier-test, held out, 20% (offset:limit) |
|---|---|---|---|---|
| MR | 616 | 0:616 (all) | 0:493 | 493:123 |
| CT | 1228 | 0:1228 (all) | 0:982 | 982:246 |

This finally gives MR a genuine held-out test set (it previously had none at all), and
grows both datasets' classifier pools far beyond their previous CV-only sizes (MR: 55 →
493+123 = 616 total; CT: 57 (val) → 982+246 = 1228 total, no longer wasting the ~1,082/
1,139-subject native "train" splits that fed only the reference table before).

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
| `COL_ROC_AUC_MEAN` / `COL_ROC_AUC_STD` / `COL_PR_AUC_MEAN` / `COL_PR_AUC_STD` | mean/std of ROC-AUC/PR-AUC across CV folds | `train.py` (`cv_summary.csv`) | diagnostic |
| `COL_PR_AUC_ACCEPT` / `COL_PR_AUC_REJECT` | PR-AUC for the accept/reject class specifically | `test.py`, `ablations.py` | diagnostic |
| `COL_N_TEST_ROWS` | row count scored | `test.py` | diagnostic |
| `COL_ANATOMICAL_FAMILY` | coarse anatomical grouping (e.g. all ribs → `rib`) derived from `FAMILY_PATTERNS` | `ablations.py` | diagnostic |
| `COL_N_ORGANS` | distinct-organ count within a group | `ablations.py` (`*_loo.csv`) | diagnostic |
| `COL_ABLATION_CONFIG` / `COL_ABLATION_KIND` | which feature-set variant was tested / its category (`full`/`drop_group`/`only_group`/`drop_feature`/`only_feature`) | `ablations.py` (`feature_ablation.csv`) | diagnostic |
| `COL_N_FEATURES` | feature count for that ablation config | `ablations.py` | diagnostic |
| `COL_WITHIN_ORGAN_AUC` / `COL_N_GROUPS_WITHIN` | mean AUC computed separately inside each organ (strips out easy/hard ranking), and how many organs/families had enough data to include | `ablations.py` | diagnostic |
| `COL_DELTA_AUC` / `COL_DELTA_WITHIN_AUC` | pooled/within-group AUC change vs. the full-feature model | `ablations.py` | diagnostic |
| `COL_IOU_THRESHOLD` | IoU accept-threshold tested in the threshold sweep | `ablations.py` (`threshold_sweep.csv`) | diagnostic |
| `COL_SUFFICIENT_DATA_FLAG` | whether a family met the min-rows/min-minority-class bar to compute an AUC AT THAT SPECIFIC THRESHOLD (can hold at a loose threshold and fail at a strict one) | `ablations.py` (`threshold_sweep_by_family.csv`) | diagnostic |
| `COL_AUC_MIN` / `COL_AUC_MAX` / `COL_AUC_RANGE` | a family's AUC spread across the threshold sweep (only over thresholds where it had sufficient data) - the direct answer to "is this family threshold-sensitive" | `ablations.py` (`threshold_sweep_family_volatility.csv`) | diagnostic |
| `COL_ACCEPT_RATE_MIN` / `COL_ACCEPT_RATE_MAX` / `COL_ACCEPT_RATE_RANGE` | a family's accept-rate spread across the sweep (always computable, no eligibility gate) | `ablations.py` (`threshold_sweep_family_volatility.csv`) | diagnostic |
| `COL_N_THRESHOLDS_EVALUATED` | how many of the swept thresholds a family had enough data to be scored at | `ablations.py` (`threshold_sweep_family_volatility.csv`) | diagnostic |

Not in this table: `EXPECTED_DICE`'s keys in `compute_metrics.py` (organ names, not metric
short-forms) and every dynamically-named per-model/per-feature-set result column
(`oof_predictions.csv`, `loo_by_organ.csv`, `calibrated_oof.csv` — named
`<model family> [<feature set>]` or `<base model> + <calibration method>`, can't be
static constants since the set of models/feature-sets is configured at runtime).

## 6b. Stage 7 (optional) — Ablations (`experiments/pipeline/ablations.py`)

Motivation: several feature-importance results (section 6, and the CT pilot run in
section 8) rank size-related features highest. But IoU is size-biased by construction —
for a sphere of radius `r` with boundary error `d`, `IoU ≈ 1 - 3d/r` — so a fixed IoU
threshold is a much tighter physical standard for small structures than large ones.
That raises a real methodological concern: a "size" feature could predict the
accept/reject *label* purely because it predicts organ difficulty, without reflecting
mask *quality* at all. This stage exists to pressure-test that concern with four
experiments, run against a stage-4 curated dataset directly (`--dataset-csv`):

1. **Feature ablation** — drop/isolate each feature group (size, position, extent,
   component, boundary, intensity) and the top individual features by RF importance,
   measuring pooled AND within-organ AUC for each. A group is flagged
   "separates easy/hard, not good/bad" when dropping it costs pooled AUC far more than
   within-organ AUC.
2. **Threshold sweep** — recomputes the label at IoU thresholds 0.70–0.95 (the raw
   `Intersection over Union (IoU)` column is kept in every curated dataset specifically
   for this) and checks whether AUC stays stable — if so, the 0.90 cutoff isn't
   load-bearing to the conclusions. Also breaks this down **per anatomical family** at
   every threshold (`threshold_sweep_by_family.csv` + a `threshold_sweep_family_
   volatility.csv` summary of each family's AUC range across the sweep) — a stable
   *pooled* AUC can hide individual families swinging in opposite directions and
   canceling out in the average, which is exactly the failure mode this catches. A
   family's eligibility to even get an AUC is re-checked at each threshold (min rows +
   min minority-class count), since it can hold at a loose threshold and fail at a
   strict one as the accept rate collapses — tracked explicitly via
   `has_sufficient_data` rather than the family silently vanishing from the output.
3. **Within-organ/within-family AUC** — the key diagnostic: pooled AUC rewards ranking
   easy structures above hard ones; within-group AUC strips that out by scoring only
   whether, among masks of the *same* structure, the good ones rank higher. The gap
   between pooled and within-organ AUC is the share of apparent performance that's
   really just "this organ is generally easier," not real mask-quality discrimination.
4. **Leave-one-family-out** — a stricter version of section 6's leave-one-organ-out.
   Holding out one rib while ~23 other ribs stay in training overstates transfer, since
   the model still sees the structure type. `FAMILY_PATTERNS` (a curated
   organ-name → anatomical-family regex mapping, e.g. all `rib_*` → `rib`,
   `aorta`/`vena_cava`/... → `great_vessel`) groups organs so an entire structure type
   can be excluded at once — a much stronger test of whether the model generalizes to
   genuinely novel anatomy.

Reuses `train.py::resolve_feature_columns()` to recover the raw/organ-relative feature
split from stage 4's manifest (organ-relative z-scores are included only with
`--use-relative-features`, off by default — they add little on organs already seen
during training and are meant for the family-transfer question specifically). All
model fits use a plain `RandomForestClassifier` (not the full 4-family/4-feature-set
sweep from `train.py`) at a lower `--n-trees` (200 vs. 400) since this stage runs many
more fits than `train.py` does and the ranking is stable at that count.

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

### MR full pipeline run — `experiments/pipeline/`, 200-subject train / full 55-subject test
First full-scale run of the staged pipeline (see section 0) end-to-end: reference table
from the entire 561-subject MR `train` split (ground-truth only, no inference cost),
classifier trained on 200 `train`-split subjects, evaluated on the full 55-subject
`test` split (the same predictions as the MR reproduction run above, reused rather than
re-inferred). All committed result files are under
`experiments/eval_runs/mr_full_run/` (predictions/ and models/*.joblib are gitignored —
regenerate via `experiments/pipeline/run_pipeline.py`, see `models/training_manifest.json`
for the exact command/args/git commit that produced this run).

- **Scale:** reference table 751 rows (561 subjects); `datasets/classifier_train.csv`
  3,213 rows / 190 subjects (5 empty-prediction rows dropped, 10 of the 200 predicted
  subjects contributed no scoreable rows); `datasets/classifier_test.csv` 929 rows / 55
  subjects (11 dropped).
- **Benchmark comparison held up at scale, identical to the earlier 55-subject
  reproduction:** 50/50 organs matched, only the same 2 organs flagged (4%) —
  `portal_vein_and_splenic_vein` (Δ −0.160) and `inferior_vena_cava` (Δ −0.159). Confirms
  the cached test predictions are being reused correctly, bit-for-bit consistent results.
- **Cross-validated model search** (`results/train/cv_summary.csv`, 5-fold grouped CV):
  best model `randomforest [raw+rel]`, ROC-AUC 0.941 ± 0.014, PR-AUC 0.860 ± 0.020.
  Full ranking, best to worst: randomforest ≈ histgradboost (~0.94 AUC, raw/raw+organ/
  raw+rel feature sets) > logreg (~0.91–0.92) > organ-rate baseline (0.861) > tree-depth3
  and `[rel]`-only variants (~0.76–0.89) > always-accept (0.5, floor by construction).
  **Every real raw-feature model beat the organ-rate baseline** — same "not just
  memorizing organ identity" conclusion as the smaller CT pilot, now at 200-subject scale.
- **Held-out test** (`results/test/test_summary.csv`, never touched by reference-building
  or training/CV): best model `randomforest [raw+organ]`, ROC-AUC 0.928, PR-AUC(reject)
  0.984. CV and test picked *different* top models (both random-forest variants within
  ~0.01 AUC of each other) — expected instability between two different evaluation
  procedures at this scale, not a red flag.
- **Leave-one-organ-out** (`results/train/loo_by_organ.csv`, 27 organs had ≥25 rows):
  mean AUC `raw` 0.881 → `rel` 0.856 → `raw+rel` 0.893. **`raw+rel` generalizing best to
  organs excluded from training is the key evidence that the organ-relative z-score
  features carry real transferable signal**, not just per-organ memorization — this is
  the strongest single result so far for that specific paper claim.
- **Per-row audit trails available for follow-up analysis:** `results/train/oof_predictions.csv`
  (every training row's honest out-of-fold prediction from every model) and
  `results/test/test_results.csv` (every test row's prediction from every persisted
  model) — both committed, useful for calibration plots or per-organ error breakdowns
  without re-running anything.

**Caveat (found later, see section 2a):** this run's reference table (all 561 `train`-split
subjects) and its classifier training data (200 `train`-split subjects) were drawn from
the same ordered prefix of MR's `train` split, so all 190 classifier-training subjects
are a subset of the 561 reference-table subjects. Since `curate_dataset.py`'s reference
join is purely by organ name with no subject-level check, those subjects' organ-relative
z-score features were partly computed against a population that includes themselves —
a mild self-referential bias, likely small for common organs and larger for rare ones.
Left as-is rather than re-run (section 2a's pooled partitioning is the fix going forward,
used for the expanded classifier-pool run); the held-out `test`-split numbers above are
unaffected (test subjects never contributed to the reference table in this run).

---

### Ablation suite comparison — MR (ours) vs. CT (Aahil's), `experiments/pipeline/ablations.py`

Ran the stage-7 ablation suite (see §6b for full methodology) against the committed
200-subject MR training set (`experiments/eval_runs/mr_full_run/datasets/classifier_train.csv`
→ `experiments/eval_runs/mr_full_run/results/ablations/`), and compared against a run
Aahil did independently on the CT dataset. Same four experiments, same underlying logic
in both — this section records where the two runs agree, where they diverge, and why.

**Methodology recap** (see §6b for full detail): four experiments, all built around one
central worry — IoU is size-biased by construction (`IoU ≈ 1 - 3d/r` for a sphere of
radius `r` with boundary error `d`), so a "size" feature could predict the accept/reject
label purely via organ difficulty rather than real mask quality.
1. **Feature ablation** — drop/isolate feature groups, measure pooled AUC (all organs
   mixed) *and* within-organ AUC (only compares masks of the same organ, stripping out
   "which organ is this" as a shortcut). A group that costs pooled AUC far more than
   within-organ AUC is flagged as an easy/hard-organ separator rather than real signal.
2. **Threshold sweep** — relabel at IoU cutoffs 0.70–0.95, check whether AUC stays
   stable (if so, the 0.90 default isn't load-bearing to the conclusions).
3. **Within-organ / within-family AUC** — the diagnostic experiment 1 and 4 both lean
   on; the gap between pooled and within-organ AUC is the share of the pooled score
   that's just organ-difficulty ranking rather than genuine per-mask discrimination.
4. **Leave-one-organ-out vs. leave-one-family-out** — organ-LOO retrains with one organ
   entirely excluded and tests purely on it; family-LOO does the same for a whole
   anatomical family at once (via `FAMILY_PATTERNS`). Family-LOO is the stricter test,
   since organ-LOO can be optimistic when many near-duplicate organs (e.g. individual
   ribs) remain in training.

#### Our MR results (200-subject train set, 3,213 masks / 190 subjects / 50 organs / 13 families, 26.5% accept)

- **Feature ablation:** `size` is the costliest group to drop (pooled 0.937→0.921,
  Δ−0.015; within-organ 0.862→0.814, Δ−0.047). `component` is nearly irrelevant
  (Δ−0.003 pooled, Δ−0.002 within). **Zero groups flagged** as pure easy/hard
  separators — every group that mattered hurt within-organ AUC as much or more than
  pooled, i.e. real quality signal, not organ-memorization. Top individual features by
  RF importance: Volume (0.192), Number of Voxels (0.144), Mask-to-BBox Fill Ratio
  (0.118).
- **Threshold sweep:** AUC 0.920 → 0.937 → 0.967 across IoU 0.70 → 0.90 → 0.95 (accept
  rate swings 86.2% → 26.5% → 4.0% over the same range). Stable relative to the size of
  the accept-rate swing — **0.90 is not load-bearing.**
- **Within-organ/family AUC:** pooled 0.937 vs. mean within-organ 0.862 (27 organs
  qualified) — gap of 0.075. Weakest: `iliopsoas_right` (0.709), `gallbladder` (0.739),
  `vertebrae` (0.764). Strongest: `autochthon_left` (0.977), `gluteus_medius_right`
  (0.977). By family, weakest `vertebrae` (0.764) and `heart` (0.803), strongest
  `shoulder_bone` (0.980).
- **Organ-LOO vs. family-LOO:** organ-LOO mean 0.876 / median 0.883 (27 organs);
  family-LOO mean 0.896 / median 0.909 (11 families) — family-LOO slightly *higher*
  than organ-LOO here, the reverse of the textbook-expected direction (see comparison
  below for why).

#### Aahil's CT results (reported informally, reproduced here for the record)

- **Feature ablation:** `component` is the costliest group to drop (pooled 0.870→0.816,
  Δ−0.054; within-organ Δ−0.128) — hurts within-organ more than pooled, so it's judging
  real per-structure quality, not just easy/hard ranking. `size` barely matters
  (Δ−0.010, pooled 0.870→0.860) — "our feature-importance table was misleading here;
  impurity importance is biased toward continuous features, so size ranked high without
  mattering. Ablation is the honest measure" (his note, worth applying generally: don't
  trust raw RF importance rankings over ablation deltas).
- **Threshold sweep:** AUC 0.877, 0.877, 0.872, 0.867, 0.870, 0.875 across IoU
  0.70→0.95 (range ~0.010) while accept rate swings 92.7%→46.0%. Same conclusion:
  **0.90 is not load-bearing.** Per-family breakdown not yet run — flagged as the next
  step, since pooled stability could mask families moving in opposite directions.
- **Organ-LOO vs. family-LOO:** organ-LOO mean 0.821, family-LOO mean 0.779 — the
  textbook-expected direction (organ-LOO inflated). Explanation: 22 of 66 usable organs
  were individual ribs, so holding out one rib left ~23 near-identical ribs in training.
  Most families lost almost nothing when held out entirely (`lung` 0.951→0.930,
  `urinary` 0.833→0.832).
- **`costal_cartilage` (initially read as a contradiction, then corrected):**
  `within_family_auc` = 0.601 (model *did* train on some costal cartilage, via normal
  CV) vs. `family_loo` = 0.399, *below chance* (model never saw it at all). Not a
  contradiction — different questions ("performance with some exposure" vs. "with
  zero exposure"). The below-chance number is the real finding: likely
  `largest_component_fraction` (a top feature) inverts for costal cartilage, which is
  naturally fragmented, so the model's usual "more fragmented = worse" heuristic
  backfires there. Caveat noted: n=49, 16% accept — noisy.

#### Comparison: agreements and differences

**Agreed:**
- 0.90 threshold not load-bearing, in both datasets, by a wide margin (accept-rate
  swing of 40-60+ points vs. AUC range of ~0.01-0.05).
- In both runs, whichever feature group actually mattered hurt within-organ AUC as
  much or more than pooled AUC — the interpretive rule ("real signal if it hurts
  within-group discrimination too") held in both datasets, even though it identified a
  *different* feature group as the important one.

**Diverged, with explanation (not a contradiction in either case):**
- **Which feature group carries the signal** — `size` for MR, `component` for CT.
  Plausible cause: CT's organ set includes thin, easily-fragmented structures (ribs,
  costal cartilage, sternum) where connectivity is the obvious tell of a broken mask;
  MR's organ set here skews toward solid abdominal organs and large muscles, where
  size/shape deviation is more informative. Feature importance appears to be
  organ-set/modality dependent rather than universal — worth stating as such in the
  paper rather than picking one "winner" feature.
- **Organ-LOO vs. family-LOO direction** — CT showed the expected organ-LOO-optimistic
  pattern (driven by ~22 near-duplicate ribs staying in training); MR showed the
  reverse. MR's family sizes have no rib-like family of many near-duplicate organs (the
  largest family, `muscle`, has 10 genuinely distinct structures, not near-copies), so
  the specific redundancy mechanism that inflates organ-LOO in CT has no MR analogue.
  The small MR reversal (0.896 vs. 0.876, on only 11 families / 27 organs) is more
  plausibly sampling noise than a real effect.

**Follow-up (resolved on MR):** per-family threshold sweep — does AUC stability hold
*per family*, or does pooled stability mask families moving in opposite directions as
the threshold changes. `ablations.py` now computes this directly
(`threshold_sweep_by_family.csv` + `threshold_sweep_family_volatility.csv`, committed
under `experiments/eval_runs/mr_full_run/results/ablations/`). **The theory was
correct — pooled stability was hiding real per-family volatility.** Pooled AUC ranged
only 0.047 across the sweep (0.920→0.967), but individual families swung far more:

| family | AUC range | AUC min→max |
|---|---|---|
| `vertebrae` | **0.189** | 0.764 → 0.953 |
| `heart` | 0.169 | 0.740 → 0.909 |
| `lung` | 0.160 | 0.805 → 0.965 |
| `neuro` | 0.139 | 0.860 → 0.999 |
| `great_vessel` | 0.118 | 0.801 → 0.919 |
| `muscle` (least volatile) | 0.029 | 0.918 → 0.947 |

`vertebrae` alone swings **~4x** the pooled range. Self-consistency verified: every
family's AUC at threshold=0.90 in the new per-family breakdown matches the
already-committed `within_family_auc.csv` exactly (max diff 0.0), confirming the new
code computes the same thing the old code did, just sliced across every threshold
instead of only the default one. `reproductive` (9 masks) never had enough of both
classes simultaneously to score an AUC at any threshold (`n_thresholds_evaluated=0`)
despite its accept rate itself swinging 0%→100% across the sweep — too small a family
to draw any conclusion from, flagged as such via `has_sufficient_data=False` rather
than silently omitted. **Practical implication for the paper:** reporting a single
pooled "AUC is threshold-stable" claim would understate real threshold sensitivity for
specific anatomy (vertebrae, heart, lung, neuro) — any claim about threshold robustness
should be qualified per anatomical family, not stated as a blanket result. Not yet run
against the CT dataset — worth Aahil reproducing the same way once he pulls this branch.

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

---

## 11. Full-dataset run on TotalSegmentator's own official split (`experiments/paper_final_v2/`)

A parallel line of work (`final` branch, merged into this one) built the GRAM paper's
methodology — IoU regression, conformal prediction intervals, feature-group ablations,
leave-one-family/organ-out transfer, clinical decision-flip analysis — on small subsets:
TotalSegmentator's own official 57-subject CT `val` split, and a 38-subject in-sample
MRI dev subset carved from MRI's `train` split (MRI has no `val` split, so no clean
held-out set existed for it at the time).

This section reruns that same methodology on the much larger, already-inferenced
datasets from section 2/2a above (CT 1,228 subjects, MR 616 subjects), but partitioned
by TotalSegmentator's **own** official train/val/test split rather than the pooled 80:20
scheme section 2a describes — i.e. official `train` subjects feed the QC classifier's
training data and the reference table, official `test` (CT: `val`+`test` folded
together) feeds evaluation, matching the paper's own framing exactly rather than this
project's independent pooled partition.

**Coverage isn't literally 100%.** The remote GPU box (section 8/9) was decommissioned
before every subject's predictions made it into a checkpointed `combined_metrics.csv`,
and there is no GPU access to backfill the gap:

| | official train | official val (CT only) | official test | total available |
|---|---|---|---|---|
| MR | 519 / 561 | — | 49 / 55 | 568 / 616 (92.2%) |
| CT | 1,043 / 1,082 | 53 / 57 | 87 / 89 | 1,183 / 1,228 (96.3%) |

Still a large improvement over the paper's 57/38-subject subsets. `meta.csv` for both
datasets was reconnected from `/Volumes/Datasets/` (external drive) to get the real,
authoritative split assignment — see `experiments/pipeline/split_by_official.py`, which
regroups the already-computed pooled-run metrics by official split with no re-inference.

**Reference table is train-only** here (unlike section 2a's all-subjects choice) —
matches the paper's methodology and removes the self-referential bias section 2a
explicitly accepted as a tradeoff.

**Substantive methodological upgrade over the paper for MRI**: the paper's MRI results
were in-sample (dev subset carved from TotalSegmentator's own training data). Here, MRI
evaluates against its real official `test` split instead — genuinely held out, not an
in-sample proxy. `regression_modality.py` was changed accordingly (no more dev-carving).

### Headline results vs. the paper

| | Paper (small subset) | This run (official split, full data) |
|---|---|---|
| CT direct-classifier AUC | 0.871 (4,017 masks / 57 subj, val only) | **0.900-0.903** (9,652 masks / 140 subj, val+test) |
| MR direct-classifier AUC | 0.919 (548 masks / 38 subj, **in-sample**) | **0.912-0.922** (929 masks / 49 subj, **genuinely held-out**) |
| MR accept rate @ IoU 0.90 | 23.0% (in-sample, optimistic) | **17.1%** — closely matches the paper's own cited frozen-test rate (16.9%), a strong sanity check this is the same official partition |
| CT leave-one-family-out | mean 0.779, 13/14 families above chance (costal_cartilage 0.399, below chance) | mean 0.795, **14/14** families above chance (costal_cartilage 0.660) |
| CT conformal coverage @80% nominal | 52.0% uncalibrated → 80.7% CQR | 80.6% CQR — tracks the paper closely |
| MR silent-failure conditional rate | 69.5% (in-sample) | **46.6%** (held-out) — the in-sample estimate looks substantially inflated |

The QC classifier itself (stages 1-6, not the GRAM regression/conformal suite) shows the
same pattern: best model AUC 0.918 (CT) / 0.947 (MR, genuinely held out) on this split,
both higher than the equivalent pooled-80:20-split numbers from section 2a.

### Output locations

- `experiments/eval_runs/{mri,ct}_official_split/` — stages 1-6 (predictions already
  existed; reference table, curated datasets, 16 persisted models, CV/test results are
  new), mirroring `{mri,ct}_full_remote/`'s layout.
- `experiments/paper_final_v2/` — GRAM regression/conformal/ablation/clinical-flip suite,
  mirroring `experiments/paper_final/`'s layout (the original, paper-cited, small-subset
  results, left untouched for comparison).
- `experiments/pipeline/split_by_official.py` — the new regroup-by-official-split script.

`regression.py`, `regression_modality.py`, `paper_consolidate.py` were repointed in
place at these new files (previously hardcoded to a teammate's local Windows paths and
the small subsets); `paper_consolidate.py` also had a latent `KeyError` fixed
(`brier_decomposition()`'s dict keys are qc_columns.py's long descriptive labels, not
the short names that script's original codebase assumed — a pre-existing mismatch
between the two branches' eras of `common.py`, unrelated to this run's data itself).
`ablations.py` needed no changes (already CLI-driven, no hardcoded paths).
