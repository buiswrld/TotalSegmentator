# Contamination audit (Phase 0)

Purpose: establish, factually, everything already computed on either held-out test split
before the rebuttal freeze, by whom and when, and whether it could have influenced any
modelling choice. Written before any new test label is read. Be factual, do not minimize.

## Verdict up front

- **CT test (89 subjects): CLEAN.** No TotalSegmentator predictions, no descriptor table,
  no IoU/Dice/accept labels for the CT test split exist anywhere in the repo or on disk.
  CT test has never been scored. It is a valid, untouched held-out set.
- **MRI test (55 subjects, 49 with scoreable masks): EXPOSED BEFORE FREEZE.** A full
  train->test reliability-model evaluation and the test accept rate were computed and
  committed on 2026-08-03, about three weeks BEFORE the paper's final MRI modelling
  decisions (2026-08-24). This cannot be undone and must be disclosed. Details below.

## What was computed on MRI test, and when

Source: `experiments/eval_runs/mr_full_run/`, committed in `0d8d690`
("Add full 200-subject MR pipeline run results") on **2026-08-03**. The manifests record
it was run from `/Users/sohamlokhande/dev/TotalSegmentator` (teammate Soham's machine),
git commit `9e9f643`, created 2026-08-03T16:58 (metrics) and 17:10 (results) UTC.

Files and what they contain (all on the FROZEN MRI test split):

| File | Content | Uses test labels? |
|---|---|---|
| `metrics/test/combined_metrics.csv` | 940 rows, 49 subjects. The 19 reference-free descriptors AND the ground-truth columns IoU, Dice, TP/FP/FN, gt_vox, Match Status, accept. | Yes: IoU/Dice/accept present. |
| `metrics/test/summary_by_organ.csv` | Per-structure mean Dice/IoU, accept rate on test. | Yes. |
| `metrics/test/expected_vs_measured.csv` | Measured vs published Dice per structure on test. | Yes. |
| `datasets/classifier_test.csv` | Curated test set (features + accept label) for scoring. | Yes (label column). |
| `results/test/test_summary.csv` | **16 reliability models scored on test**: ROC-AUC, PR-AUC(accept/reject), ECE, MCE, Brier. Best randomforest [raw+organ] ROC-AUC 0.928; every model and feature set ranked on test. | Yes. |
| `results/test/test_results.csv` | Per-mask predicted probabilities + true label on test. | Yes. |

So as of 2026-08-03 the team could see, on the MRI test split: the full reliability-model
leaderboard (classifier, four feature sets, ROC-AUC up to 0.928, PR-AUC, calibration
error), the per-structure Dice/accept table, and the overall test accept rate.

## The 16.9% MRI test accept rate

Confirmed origin: the MRI test accept rate reported in the paper (16.9%) is the
IoU >= 0.90 rate over the 929 non-empty masks in `metrics/test/combined_metrics.csv`,
computed on 2026-08-03 as part of the run above (`iou_accept 0.9` in its manifest). It is
a test-label-derived statistic that was read before the freeze. It is a dataset property
(prevalence), not itself a model choice, but it confirms test labels were inspected.

## Timeline and influence assessment

- **2026-08-03**: MRI test scored end to end (classifier, train->test), committed. Test
  AUC leaderboard and accept rate visible in-repo.
- **2026-08-24**: the paper's FINAL MRI modelling decisions were committed
  (`6a34e2f`, `regression_modality.py` / `paper_consolidate.py`): carve a 20% dev subset
  from MRI train, use an IoU **regressor** (not the classifier) as the MRI primary, the
  with/without-intensity split, and the decision to report MRI **uncalibrated** (isotonic
  and Platt both failed on dev).

Factual assessment, neither inflating nor minimizing:

1. The **specific frozen MRI models were never selected against test**. The Aug-03 run
   evaluated the **classifier** family; the paper's MRI primary is a **random-forest IoU
   regressor**, which does not appear in `test_summary.csv`. The regressor has not been
   scored on test until this rebuttal.
2. However, **aggregate MRI test performance (classifier ROC-AUC up to 0.928) and the test
   accept rate were visible during the window** in which the regression-vs-classifier,
   intensity, and calibration decisions were finalized. We cannot prove these decisions
   were free of that knowledge. The honest position is that MRI test is a **previously
   exposed** split, not a clean held-out evaluation for model-selection purposes.
3. MRI test is **also in-sample for the segmenter** (MRI has no official val split; the
   whole MRI protocol is a secondary in-sample check). So MRI test carries two caveats:
   previously exposed to the team, and in-sample for TotalSegmentator.

## CT test: evidence of cleanliness

- No CT test prediction directory exists. `predictions_ct_val/` is the 57 dev (val)
  subjects; `predictions_ct/` (20) is the CT pilot; `predictions/` (5) is a small set;
  none intersect the 89 CT test subjects. `Algoverse/test_out/` is a single stray
  TotalSegmentator run (loose per-structure files, no subject folders), not the test set.
- No CT test `combined_metrics` or any CT-test metrics/label file exists in the repo or in
  `Algoverse/`. The only `compute_metrics` run with test split on record is MRI
  (dataset-dir = the MRI dataset).
- CT test images and reference masks are present on disk (89/89), so predictions can be
  generated fresh in Phase 2 without ever having read a CT test label.

## MANIFEST C6 context (does not prove the test numbers were unseen)

`experiments/paper_final/MANIFEST.md` item **C6** flags `experiments/eval_runs/mr_full_run/`
as a *different* MRI protocol (MRI train->test full-pipeline run, 190->49 subjects) and
instructs: "`test_summary.csv` is on the **FROZEN MRI test** - do NOT cite as a dev result
... Keep separate from the dev-subset MRI numbers. Do not mix."

State this plainly: C6 is an instruction not to *cite or mix* the mr_full_run test numbers
with the dev-subset numbers during drafting. It is evidence the team knew these were test
results and tried to wall them off. It does **not** prove the test numbers were unseen when
the final MRI modelling choices (`6a34e2f`, 2026-08-24) were made. The files sat in the repo
from 2026-08-03 onward, and a "do not cite" note constrains the write-up, not what a person
had already seen. MRI test remains a previously exposed split.

## Pre-freeze facts resolved before the freeze (rebuttal A1, A2)

**A1 - MRI dataset splits (reference-free, from `meta.csv`).** The MRI dataset `meta.csv`
(`TotalsegmentatorMRI_dataset_v200`, 616 subjects) has exactly two split values:
**train = 561, test = 55. There is no validation split.** (CT `meta.csv`, 1228 subjects:
train = 1082, test = 89, val = 57.) No official MRI validation split exists, so the freeze
proceeds. The paper's "190 train / 49 test" is the `mr_full_run` processed subset, not the
full dataset: `mr_full_run/train` scored 190 of the 561 official train subjects
(range s0001-s0444), the 38-subject dev set is the subject-grouped 20% carve of those 190,
and `mr_full_run/test` scored 49 of the 55 official test subjects.

**A2 - the 6 MRI test subjects with zero rows.** Official MRI test = 55; `mr_full_run/test`
`combined_metrics.csv` has 49 subjects. The 6 with zero rows are
**s0078, s0150, s0458, s0480, s0523, s0580**. Reference-only diagnosis:

| Subject | Dir | `mri.nii.gz` | Ref masks | Finding |
|---|---|---|---|---|
| s0078 | present | 7.8 MB, readable | 50/50 present | image + all references present |
| s0150 | present | 8.1 MB, readable | 50/50 present | image + all references present |
| s0458 | present | 2.2 MB, readable | 50/50 present | image + all references present |
| s0480 | present | 2.0 MB, readable | 50/50 present | image + all references present |
| s0523 | present | 11.5 MB, readable | 50/50 present | image + all references present |
| s0580 | present | 2.7 MB, readable | 50/50 present | image + all references present |

All 6 have a subject directory, a readable non-zero `mri.nii.gz`, and all 50 reference
masks on the canonical v200 dataset. So the cause is **not** unreadable input and **not**
missing reference structures.

`compute_metrics.py` (lines 383-385) skips a subject and emits zero rows the moment its
`predictions/<subject>/` directory contains no `*.nii.gz` ("no predictions ... skipping
(run stage 1 first)"), and this check runs **before** any reference mask is loaded, so
reference content cannot rescue the subject. The predicted masks for `mr_full_run` were
written to `experiments/eval_runs/mr_full_run/predictions/test` on the teammate's machine
and were **never committed** (no `predictions/` dir in the repo), and **no stage-1 inference
log/manifest for `mr_full_run` was committed** either. Conclusion: these 6 subjects were
dropped because upstream stage-1 inference produced no predicted masks for them; from the
committed artifacts alone it cannot be determined whether stage-1 crashed on them or emitted
empty output. Under the frozen rule (config B2), all 55 official MRI test subjects are
attempted in Phase 2; a subject whose regenerated predictions come back all-empty is kept
under the empty-mask rule, and only a reproducible crash / unreadable input would exclude it.

## Implications for the rebuttal

- **CT held-out test results will be a genuine first-time evaluation.** Report them as
  such.
- **MRI held-out test results must be reported with both caveats stated plainly**: the MRI
  test split was already scored by a teammate on 2026-08-03 (before the final MRI model
  choices), and it is in-sample for the segmenter. Re-running the frozen MRI regressor on
  test is still informative (the regressor itself was not tuned on test), but the paper
  should not describe MRI test as untouched or as a clean generalization test.
- Recommendation to surface to the mentor: decide how to frame MRI test in the paper
  (disclose the prior exposure explicitly), and treat CT test as the primary held-out
  evidence.
