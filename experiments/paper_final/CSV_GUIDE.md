# GRAM — plain-language guide to the final CSVs

**What this folder is.** `paper_final/` holds every results table behind the GRAM paper
(reference-free per-mask reliability for TotalSegmentator). GRAM looks at a single
predicted segmentation mask — with **no ground truth** — and scores how likely that mask
is good. To build and check that scorer we *did* compare against ground truth, but only to
create labels; the scorer itself never sees ground truth at prediction time.

**Two datasets, two very different footings.**
- **CT** files (`ct/`) come from the **validation split**, which is **genuinely held out** —
  TotalSegmentator never trained on these scans. These are the headline results.
- **MRI** files (`mri/`) come from a **20% "dev" subset carved out of the MRI training set**
  (split by patient, so no patient is on both sides). ⚠ **Every MRI number is in-sample:**
  the segmenter was trained on these scans, so MRI is a *secondary sanity check that the
  method carries over*, **not** a clean held-out replication like CT. Say "in-sample"
  wherever MRI numbers appear. MRI test was never touched.

**A few words used throughout.**
- **accept label** = a mask is "accept" if its overlap with ground truth (IoU) is ≥ 0.90,
  otherwise "reject." This is the thing GRAM tries to predict.
- **OOF** ("out-of-fold") = every mask is scored by a model that never saw that patient
  during training, so the numbers aren't inflated by memorization.
- **AUC** = how well the score ranks good masks above bad ones (1.0 perfect, 0.5 coin-flip).
- On **CT the minority (rare) class is "reject"** (most masks are good); on **MRI the minority
  is "accept"** (most MRI masks fall below the 0.90 bar). The rare class is the hard,
  important one, so watch which one it is.

Where a CT file and an MRI file are a **matched pair**, it's called out so you can compare.

---

## CT — `ct/` (held-out validation split)

### Dataset

**`dataset_combined_metrics.csv`** — The raw table everything else is built from: one row
per (patient, structure), 4033 rows. Columns are the accuracy labels from comparing to
ground truth (Dice, IoU, accept, etc.) plus ~20 "reference-free" mask descriptors (size,
shape, location, connected pieces, boundary contact, intensity). Backs the dataset
paragraph / Table 1. Headline: **4033 masks, 16 empty → 4017 modelled, 57 patients, 117
structures, 74.5% accept at IoU 0.90.** Matched pair: `mri/dataset_combined_metrics.csv`.

**`dataset_summary_by_organ.csv`** — The same data rolled up to one row per structure
(117 rows): how many were scored, average Dice/IoU, and the per-structure accept rate.
Answers "which organs does TotalSegmentator do well or badly on." Caveat: its accept rate
(≈75.2%) uses a slightly narrower denominator than the 74.5% headline — it drops 38
"false-positive" masks (predicted where there was no organ) — so the small difference is
expected, not an error. Matched pair: `mri/dataset_summary_by_organ.csv`.

**`dataset_accept_analysis.csv`** — One row per structure showing how many masks were
accepted and the average IoU of the accepted vs. rejected ones. It's a supporting sanity
view of how cleanly the 0.90 bar separates good from bad per organ. No single headline
number; supplementary. (CT-only — no MRI counterpart.)

### Primary model performance

**`primary_model_oof.csv`** — The per-mask scores themselves: for every mask (4017 rows),
the out-of-fold predicted accept-probability from each model tried (logistic regression,
small tree, random forest, gradient boosting × four feature sets), plus two baselines
(organ-rate and always-accept). This is the *source* for the summary below; you'd cite the
summary, not this. Backs the primary-result figure.

**`primary_model_summary.csv`** — The headline model table: for each model, its AUC with a
95% confidence interval (from resampling patients 1000×) and precision on each class. This
answers the paper's central question — does a geometry-based scorer beat just knowing which
organ it is? Headline: **random forest AUC 0.871 [0.852, 0.892]** vs. the **organ-rate
baseline 0.761 [0.737, 0.785]** — the intervals don't overlap, so the model genuinely beats
"organ identity," not just memorizes easy organs. Matched pair: `mri/primary_model_summary.csv`.

### Ablations (which features carry the signal)

**`feature_ablation.csv`** — Drops one feature group at a time (and tries each alone), 25
rows, reporting AUC overall and *within* each organ. Answers "which kinds of features
matter, and do they judge mask quality or just tell organs apart." Headline: **full-model
AUC 0.870; the connected-components/shape features dominate, and raw size is nearly
redundant (removing it costs only 0.010 AUC).** Backs the ablation table. (CT-only.)

**`within_organ_auc.csv`** / **`within_family_auc.csv`** — How well the scorer separates
good from bad masks *inside* a single structure (66 organs) or organ family (13). This is
the strict test: it can't cheat by ranking easy organs above hard ones. Headline: **mean
within-organ AUC 0.762.** Worst is duodenum (~0.25). Supplementary to the ablation. (CT-only.)

### Threshold robustness

**`threshold_sweep_classifier.csv`** — Re-runs the classifier defining "good" at IoU cutoffs
from 0.70 to 0.95 (6 rows), with overall and within-organ AUC. Answers "is our 0.90 choice
special?" Headline: **AUC stays ~0.87 across all cutoffs → 0.90 is not load-bearing.** (CT-only
in classifier form.)

**`threshold_sweep_regression.csv`** — Same idea but from a single model that predicts the
IoU number directly, then applies each cutoff (5 rows). Shows one model can serve every
threshold. Headline: **AUC 0.886 (at 0.70) → 0.856 (0.90) → 0.838 (0.95).** Matched pair:
`mri/threshold_sweep_regression.csv`.

**`iou_regression_oof.csv`** — The per-mask predicted IoU (0–1) from two models, 4017 rows.
Feeds the threshold-regression view and the conformal intervals. Headline: **random-forest
Spearman 0.693 with true IoU, R² 0.427.** Matched pair: `mri/iou_regression_oof.csv`.

### Transfer to unseen structures

**`transfer_leave_one_organ.csv`** — Trains with one organ entirely removed, tests on it
(66 organs). Answers "does the scorer work on a structure it never trained on?" Headline:
**mean AUC 0.821, median 0.872.** Backs the transfer table. Note: a second, older
leave-one-organ file exists elsewhere (columns raw/rel/raw+rel, mean 0.836) — that one
tests *feature-set* transfer and is a different question; use this file for the
organ-vs-family story. (CT-only.)

**`transfer_leave_one_family.csv`** — The harder version: removes a whole organ *family*
(13 families) so no lookalike remains. Headline: **mean AUC 0.779, median 0.829** — the drop
from the organ-level number is the honest cost of true transfer, not a bug. Pairs with the
file above. (CT-only.)

### Calibration (are the scores trustworthy probabilities?)

**`calibration_metrics.csv`** — 6 rows: random forest (weighted/unweighted) × three
calibration methods (none / isotonic / Platt), each with AUC and calibration-error numbers
(ECE, MCE, Brier). AUC says the ranking is good; ECE says whether "0.8" really means 80%.
Headline: **isotonic calibration cuts ECE to 0.013 from 0.029 uncalibrated.** Backs the
calibration table. Matched pair (different result!): `mri/calibration_metrics.csv`.

**`calibration_oof.csv`** — The per-mask calibrated probabilities (4017 rows) behind the
reliability diagram, one column per model/calibration combination. Supplementary; backs the
calibration figure. (CT-only.)

### Conformal prediction intervals (a range, not just a point)

**`conformal_summary.csv`** — For each method and target coverage (80% and 90%), how often
the true IoU actually landed inside the predicted interval, plus interval width (10 rows).
Answers "can we put an honest range around each IoU prediction?" Headline: **raw intervals
covered only 52.0% at the 80% target; after conformal correction, 80.7% (median width 0.091)
at 80% and 90.7% at 90%.** ⚠ Cite the **52.0%** uncalibrated number from here — an older
file elsewhere says 55.3% from a superseded method; 52.0 is the correct one. Backs the
conformal table. Matched pair: `mri/conformal_summary.csv`.

**`conformal_by_family.csv`** — The same coverage broken out per organ family (15 rows),
with how many masks were available to calibrate each. Supplementary; shows where coverage is
thin. Matched pair: `mri/conformal_by_family.csv`.

**`conformal_intervals_oof.csv`** — The actual per-mask interval bounds (4017 rows: raw,
corrected, and family-specific lower/upper). Backs the conformal figure; also shows wider
intervals line up with bigger errors (correlation ≈ 0.46). Matched pair:
`mri/conformal_intervals_oof.csv`.

### Volume error and the "silent failure" problem

**`volume_label_oof.csv`** — Per mask (3979 rows), everything needed to study a
clinically-flavored alternative label: is the predicted *volume* off by >10% from the true
volume? Columns add `vol_err`, `vol_bad`, and `iou_reject` to the raw table. Answers "can we
predict volume error, and does a volume check catch the same failures as IoU?" Headline:
**predicting bad-volume AUC 0.825; but 52.8% of the masks that actually failed (IoU<0.90)
had near-correct volume** — i.e. a volume-only check would wave them through. Backs the
volume-label table. Matched pair: `mri/volume_label_oof.csv`. (See also
`../silent_failure_summary.csv`.)

**`volume_label_ablation.csv`** — Which feature groups drive the *volume* label (12 rows).
Companion to the volume-label result. Headline: **same as the IoU label — shape/components
dominate, size is redundant (drop −0.010).** Matched pair (and a key contrast!):
`mri/volume_label_ablation.csv`.

### Clinical impact

**`clinical_decision_flips.csv`** — For spleens (42 rows), does the predicted volume land on
the *opposite side* of the splenomegaly cutoff (314.5 mL) from the true volume — i.e. would
the mask flip the clinical call? Answers "do mask errors change a real decision?" Headline:
**0 flips out of 42 — spleen volumes are decision-robust here** (an honest null result).
Cutoff source: Prassopoulos et al., *Eur Radiol* 1997 (PMID 9038125). Matched pair:
`mri/clinical_decision_flips.csv`.

### Motivation

**`size_tolerance.csv`** — One row per organ (115) showing how much boundary error, in mm
and in voxels, the "IoU ≥ 0.90" bar actually allows for a typical organ of that size.
Explains *why* the accept bar is unfair to small structures. Headline: **liver allows
2.31 mm (1.54 voxels) of slack, but the adrenal gland only 0.35 mm (0.23 voxels).** Backs
the motivation figure. Matched pair: `mri/size_tolerance.csv`.

---

## MRI — `mri/` (in-sample dev subset — ⚠ read every number as in-sample)

Same columns and meaning as their CT twins unless noted. The MRI dev subset is **548 masks,
38 patients, 50 structures, 23.0% accept** (accept is the *minority* class here). Files
ending **`_no_intensity`** repeat the analysis with the intensity features removed, because
**MRI intensity values are not standardized across scans and are unreliable** — always
report both and treat intensity as untrustworthy on MRI. The six intensity-removed twins are
`calibration_metrics_no_intensity.csv`, `conformal_summary_no_intensity.csv`,
`conformal_by_family_no_intensity.csv`, `threshold_sweep_regression_no_intensity.csv`,
`volume_label_ablation_no_intensity.csv`, and `volume_label_oof_no_intensity.csv`; each is
identical in format to the base file described below, just with the five intensity columns
(`mean_HU/median_HU/std_HU/p05_HU/p95_HU`) excluded from the model.

### Dataset

**`dataset_combined_metrics.csv`** — The MRI dev subset raw table, one row per
(patient, structure), 550 rows, same columns as the CT version. Headline: **550 masks, 2
empty → 548 modelled, 38 patients, 50 structures, 23.0% accept.** Pairs with
`ct/dataset_combined_metrics.csv`. ⚠ in-sample.

**`dataset_summary_by_organ.csv`** — Per-structure rollup (50 rows): scored count, mean
Dice/IoU, accept rate. Note MRI covers fewer structures than CT and has whole-organ labels
(e.g. one "lung," not lobes). Pairs with `ct/dataset_summary_by_organ.csv`. ⚠ in-sample.

### Primary model performance

**`primary_model_summary.csv`** — The MRI version of the headline model table (3 rows: two
reliability scores + the organ-rate baseline), each with AUC, 95% patient-bootstrap CI, and
per-class precision. Headline: **reliability model AUC 0.919 [0.890, 0.947] vs. organ-rate
baseline 0.815 [0.766, 0.861] — intervals don't overlap**, so the "beats organ identity"
result holds on MRI too. Pairs with `ct/primary_model_summary.csv`. ⚠ in-sample; note the
MRI baseline (0.815) is higher than CT's because MRI accept rates vary more sharply by organ.

### Calibration

**`calibration_metrics.csv`** (+ **`_no_intensity`**) — MRI version, 3 rows (no /
isotonic / Platt). ⚠ **Key difference from CT: calibration does NOT help on MRI** — with
intensity both isotonic (0.054) and Platt (0.061) make ECE *worse* than uncalibrated (0.036)
and cost some AUC, because the dev set is too small for calibration to fit reliably.
Headline to report: **use MRI uncalibrated; we tried both isotonic and Platt and neither
survived.** Pairs with (and deliberately contrasts) `ct/calibration_metrics.csv`.

### Threshold and regression

**`iou_regression_oof.csv`** — Per-mask predicted IoU, 548 rows. Headline: **random-forest
Spearman 0.807, R² 0.604** (higher than CT — MRI IoU is more spread out and easier to rank).
Pairs with `ct/iou_regression_oof.csv`. ⚠ in-sample.

**`threshold_sweep_regression.csv`** (+ **`_no_intensity`**) — One IoU model applied at each
cutoff, 5 rows. Headline: **AUC rises from 0.887 (0.70) to 0.978 (0.95).** Pairs with
`ct/threshold_sweep_regression.csv`. Caveat: its `pr_auc_reject` column is precision on the
*reject* class, which on MRI is the *majority* class and therefore not the interesting one —
for MRI look at the accept-class precision in `primary_model_summary.csv` instead.

### Conformal

**`conformal_summary.csv`** (+ **`_no_intensity`**) — Coverage and width per method/target,
10 rows. Headline: **raw 45.4% → conformal 78.6% at the 80% target (median width 0.158),
87.0% at 90%.** Pairs with `ct/conformal_summary.csv`. Wider intervals than CT — MRI IoU is
harder to pin down. ⚠ in-sample.

**`conformal_by_family.csv`** (+ **`_no_intensity`**) — Per-family coverage and
calibration-set sizes, 12 rows. Caveat: the per-family conformal variant is noisy at this
small scale — report it, don't lean on it. Pairs with `ct/conformal_by_family.csv`.

**`conformal_intervals_oof.csv`** — Per-mask interval bounds, 548 rows. Pairs with
`ct/conformal_intervals_oof.csv`. ⚠ in-sample.

### Volume error

**`volume_label_oof.csv`** (+ **`_no_intensity`**) — Per mask (516 rows), the volume-error
label and features. Headline: **bad-volume AUC 0.839; and 69.5% of masks that failed on IoU
had near-correct volume** — the "silent failure" gap is even bigger than CT. Pairs with
`ct/volume_label_oof.csv`. ⚠ in-sample; raw silent-failure rate 52.5% is inflated by MRI's
high reject rate — compare the *conditional* 69.5% to CT's 52.8%.

**`volume_label_ablation.csv`** (+ **`_no_intensity`**) — Which features drive the volume
label on MRI (12 / 10 rows). ⚠ **Notable contrast with CT: here size becomes the dominant
feature (dropping it costs −0.050) while shape/components collapse** — the important
features flip between the two labels. Pairs with `ct/volume_label_ablation.csv`.

### Clinical

**`clinical_decision_flips.csv`** — Spleen splenomegaly-cutoff flips, 14 rows. Headline:
**1 flip out of 14 — anecdotal only** (far too few spleens in the dev subset to conclude
anything, and the one flip wasn't caught by the filter). Pairs with
`ct/clinical_decision_flips.csv`. ⚠ tiny sample + in-sample.

### Motivation

**`size_tolerance.csv`** — Per-organ allowed boundary error recomputed from MRI voxel
spacing (50 rows). Headline: **voxel tolerances are even tighter than CT (e.g. adrenal 0.12
voxels)** — which helps explain MRI's much lower accept rate. Pairs with
`ct/size_tolerance.csv`. ⚠ in-sample.

---

## Root files (span both modalities)

**`primary_model_comparison.csv`** — 3 rows (CT, MRI-with-intensity, MRI-without) side by
side: accept rate, the direct-classifier AUC (with and without class weighting), per-class
precision, and which class is the minority. A quick CT-vs-MRI scoreboard. Note: on MRI,
class weighting barely matters (0.910 vs 0.908).

**`silent_failure_summary.csv`** — The headline "a volume check isn't enough" number, made
comparable across modalities. Two rates per modality: the **raw rate** (of all masks, how
many are correct-volume-but-wrong-IoU) and the **conditional rate** (of the masks that
actually failed, how many had near-correct volume). Cite the **conditional** rate for a fair
comparison: **CT 52.8% vs MRI 69.5%** — in both, a volume-only check misses *most*
spatially-wrong masks. The MRI raw rate (52.5%) looks alarming but is inflated by MRI's high
reject base rate, and is ⚠ in-sample.

---

*Anything genuinely ambiguous?* The two `*_oof.csv` families (`primary_model_oof`,
`iou_regression_oof`, `calibration_oof`, `conformal_intervals_oof`, `volume_label_oof`) are
per-mask working files — useful for re-deriving a number or drawing a figure, but you'd cite
the matching summary table, not these directly. Everything else maps cleanly to a paper
section as listed. For deeper provenance (which run produced each file, and the
conflicts-to-avoid), see `MANIFEST.md`.
