# GRAM paper — final data manifest

Reference-free per-mask reliability for TotalSegmentator (CT + MRI). This folder holds
**copies** of the authoritative result CSVs (originals untouched). CT = val split
(genuinely held out). MRI = a subject-grouped 20% dev subset carved from MRI **train**.

> **⚠ Every MRI number is IN-SAMPLE for TotalSegmentator.** MRI train is the segmenter's
> own training data, so the MRI dev subset is a *secondary in-sample generalization check*,
> not a clean held-out replication like CT val. State this wherever MRI results appear.

---

## ⚠ GAPS AND CONFLICTS — READ FIRST

### Conflicts (numbers that disagree across files — decide before citing)

| # | Issue | Authoritative value / file | Stale/other value / file | Action |
|---|---|---|---|---|
| C1 | **Uncalibrated conformal coverage @80%** (the old 52.0 vs 55.3) | **52.0%** — `ct/conformal_summary.csv`, row `uncalibrated (raw quantiles)`, α=0.2. Same quantile fits as CQR, so internally consistent. | **55.3%** — `report_ct_val/regression/quantile_intervals_oof.csv` (older standalone Task-2 protocol, full-training fits). NOT copied here. | Cite **52.0%**. Make sure the drafted methods do not still say 55.3%. |
| C2 | **CT accept rate @ IoU 0.90** | **74.5%** (2993/4017 non-empty) — `ct/dataset_combined_metrics.csv`, `ct/threshold_sweep_*`. | **75.2%** (2992/3979) — `ct/dataset_summary_by_organ.csv`, whose denominator excludes the 38 `gt_vox==0` false-positive rows. | Not a bug — definitional. Cite **74.5%** as headline; if you quote summary_by_organ note its scored-only basis. |
| C3 | **Two CT regression dirs** | `report_ct_val/regression_phase2/` (Aug 24) → copied here. Adds `size_tolerance.csv`; Mondrian uses the feasibility floor (n≥4/≥9). | `report_ct_val/regression/` (Aug 17, superseded). Identical to phase2 to 1e-16 **except** Mondrian per-family (fixed n≥50 floor) and no size_tolerance. | Cite phase2 (copied). Do not cite the old Mondrian per-family numbers. |
| C4 | **Two CT leave-one-organ-out files** | `ct/transfer_leave_one_organ.csv` (ablations; single `auc`, mean **0.821**) — matched methodology to leave-one-family-out, the correct LOO/LOFO pair. | `report_ct_val/loo_by_organ.csv` (raw/rel/raw+rel columns, raw+rel mean **0.836**) — a *feature-set* transfer test, different question. NOT copied. | Use the copied file for the LOO-vs-LOFO story; cite the old one only if you specifically discuss feature-set transfer. |
| C5 | **Stale run dirs** | — | `report_ct/` (20-subject CT pilot, `combined_metrics` 1463 rows + `reference_stats`), `report/` (Jul-13 old MRI harness). | Ignore for results. `report_ct/reference_stats.csv` is a pipeline **input** (z-score table), not a result. |
| C6 | **`experiments/eval_runs/mr_full_run/`** is a *different* MRI protocol | — | MRI **train→test** full-pipeline run (long schema, 190→49 subjects). `test_summary.csv` is on the **FROZEN MRI test** — do NOT cite as a dev result; `cv_summary.csv`/`loo_by_organ.csv` are MRI train (in-sample, different scale than the dev subset). | Keep separate from the dev-subset MRI numbers. Do not mix. |

### Gaps — status after the consolidation pass

Regenerated via `experiments/pipeline/paper_consolidate.py` (no retraining for G7; G1 is a
subset; G6 fits only on the MRI dev subset — test never touched). G3/G4/G5 deliberately
skipped: MRI is scoped as an in-sample generalization probe, not a full replication.

| # | Item | Status | File(s) |
|---|---|---|---|
| G7 | CT primary-model AUC + 95% **subject-bootstrap CI** | **RESOLVED** | `ct/primary_model_summary.csv` |
| G1 | MRI-dev dataset stats + **per-structure accept rate** | **RESOLVED** | `mri/dataset_combined_metrics.csv`, `mri/dataset_summary_by_organ.csv` |
| G6 | MRI-dev calibration (ECE/MCE/Brier, none vs isotonic) | **RESOLVED** | `mri/calibration_metrics.csv` (+`_no_intensity`) |
| G2 | MRI-dev primary-model performance + organ-rate baseline | **RESOLVED** | `mri/primary_model_summary.csv`: reliability-score AUC+bootstrap CI + both-class PR **and** the subject-grouped organ-rate baseline. **Model 0.919 [0.890, 0.947] vs baseline 0.815 [0.766, 0.861] — CIs disjoint**, so the reject-vs-memorization claim holds on MRI (in-sample) exactly as on CT. |
| G3 | MRI-dev IoU-label feature-group ablation | **SCOPED OUT** | CT-depth analysis; cheap to add later if a reviewer asks. |
| G4 | MRI-dev classifier threshold sweep 0.70–0.95 | **SCOPED OUT** | MRI has the IoU-regression sweep; classifier sweep skipped. |
| G5 | MRI-dev transfer (LOO / LOFO) | **SCOPED OUT** | In-sample; skipped. |

**Net:** CT complete for all 9 groups (now with AUC CIs). MRI-dev covers 1, 2, 4(regression),
6, 7, 8, 9 + clinical/size-tolerance; only 3, 4-classifier, 5 intentionally out of scope.

> **G6 caveat:** on the small (548-mask) in-sample MRI dev set, **neither isotonic nor Platt
> (sigmoid) calibration reliably helps** — with intensity both worsen ECE (none 0.036 →
> isotonic 0.054, sigmoid 0.061) and drop AUC ~0.03; without intensity isotonic marginally
> helps ECE (0.041→0.035) but Platt worsens it (0.069), and both still cost AUC. Both
> calibrators overfit the tiny nested calibration split. **Report MRI uncalibrated** as the
> operating point (we tried both, neither survived); contrast CT, where isotonic clearly
> helped (0.029→0.013).

---

## CT — `ct/` (val split, genuinely held out)

| File | What it is | Paper section | Split | Rows | Headline number(s) to cite |
|---|---|---|---|---|---|
| `dataset_combined_metrics.csv` | One row per (subject, structure): GT-derived labels + reference-free descriptors | Dataset / Table 1 | val | 4033 | 4033 masks, 16 empty → 4017 modelled, 57 subj, 117 structures; **accept@0.90 = 74.5%** |
| `dataset_summary_by_organ.csv` | Per-structure scored count, mean Dice/IoU, accept rate | Table 1 / per-organ | val | 117 | per-structure accept rate (denominator = scored, see C2) |
| `dataset_accept_analysis.csv` | Accept breakdown + IoU of accepted/rejected per organ | Dataset supp. | val | 117 | — |
| `primary_model_oof.csv` | OOF preds for all model×featureset + organ-rate baseline + always-accept | Primary result / Fig 2 | val | 4017 | (source for the summary below) |
| `primary_model_summary.csv` | Per-model OOF **AUC + 95% subject-bootstrap CI** (1000 grouped draws) + PR-AUC both classes + minority | Primary result / Table 2 | val | 18 | **RF AUC 0.871 [0.852, 0.892]**, PR-AUC(reject, minority) 0.731; **organ-rate baseline 0.761 [0.737, 0.785]** → model's CI clears the baseline; always-accept ≈ 0.5 |
| `feature_ablation.csv` | Leave-one-group-out & only-one-group, pooled + within-organ AUC deltas | Ablation / Table 2 | val | 25 | full AUC 0.870; component dominant, size redundant (drop −0.010) |
| `within_organ_auc.csv` / `within_family_auc.csv` | Per-organ / per-family within AUC | Ablation supp. | val | 66 / 13 | within-organ mean 0.762 |
| `threshold_sweep_classifier.csv` | Classifier AUC/within-organ across IoU 0.70–0.95 | Threshold robustness | val | 6 | AUC ~0.87 flat → 0.90 not load-bearing |
| `threshold_sweep_regression.csv` | One IoU-regression model, binary metrics at each cutoff | Threshold / one-model | val | 5 | AUC 0.886(0.70)→0.856(0.90)→0.838(0.95) |
| `iou_regression_oof.csv` | OOF predicted IoU (HistGB, RF) | Regression / Fig | val | 4017 | RF Spearman 0.693, R² 0.427; regress@0.90 AUC 0.856 (vs direct 0.870) |
| `transfer_leave_one_organ.csv` | Leave-one-organ-out AUC | Transfer / Table 3 | val | 66 | **mean 0.821, median 0.872** |
| `transfer_leave_one_family.csv` | Leave-one-family-out AUC | Transfer / Table 3 | val | 13 | **mean 0.779, median 0.829** (drop vs LOO = honest transfer cost) |
| `calibration_metrics.csv` | AUC/ECE/MCE/Brier decomp, RF × {balanced,none} × {none,isotonic,sigmoid} | Calibration / Table 4 | val | 6 | **isotonic ECE 0.0129** vs uncalibrated 0.0286; Brier ~0.118 |
| `calibration_oof.csv` | Per-mask calibrated probabilities | Calibration / reliability diagram | val | 4017 | — |
| `conformal_summary.csv` | Coverage/width/Spearman: uncalibrated, CQR, Mondrian, ±clip, @80/90 | Conformal / Table 5 | val | 10 | **uncal 52.0% → CQR 80.7%@80 (medW 0.091), 90.7%@90 (0.145)**; Mondrian 81.8% |
| `conformal_by_family.csv` | Per-family coverage + calibration-set sizes (feasibility floor) | Conformal supp. | val | 15 | — |
| `conformal_intervals_oof.csv` | Per-mask raw/CQR/Mondrian bounds | Conformal / Fig | val | 4017 | width-vs-error Spearman ≈ 0.46 |
| `volume_label_oof.csv` | Per-mask vol_err, vol_bad, iou, features | Volume label / Table 6 | val | 3979 | AUC 0.825, PR(bad) 0.430, ECE 0.028; **silent-failure raw 13.1%, conditional 52.8%** (see `../silent_failure_summary.csv`) |
| `volume_label_ablation.csv` | Feature-group ablation on the volume label | Volume label | val | 12 | component dominant, size redundant (drop −0.010) |
| `clinical_decision_flips.csv` | Splenomegaly decision-flip analysis | Clinical / Table 7 | val | 42 | **0 flips / 42** (honest null; spleen volumetry decision-robust) |
| `size_tolerance.csv` | Per-organ IoU-0.90 boundary tolerance (mm & voxels) | Motivation / Fig 1 | val | 115 | liver 2.31mm/1.54vox … adrenal 0.35mm/0.23vox |

Clinical cutoff citation (spleen): splenomegaly volume > **314.5 mL**, Prassopoulos et al., *Eur Radiol* 1997;7(2):246–248 (PMID 9038125). Liver/kidney dropped (no applicable fixed, sex-independent, per-organ cutoff).

---

## MRI — `mri/` (dev subset carved from MRI train — ⚠ IN-SAMPLE)

Primary variant = **with intensity**; `*_no_intensity.csv` = intensity group dropped
(MRI intensity is uncalibrated/per-scan arbitrary — report both, flag intensity as
unreliable). Dev subset: **548 masks, 38 subjects, 50 structures, accept@0.90 = 23.0%**
(accept is the **minority** class on MRI).

| File | What it is | Paper section | Split | Rows | Headline number(s) |
|---|---|---|---|---|---|
| `dataset_combined_metrics.csv` | MRI dev subset, one row per (subject, structure) — carved exactly as the runs (G1) | Dataset (MRI) / Table 1 | dev | 550 | 550 masks, 2 empty → 548 modelled, 38 subj, 50 structures; **accept@0.90 = 23.0%** |
| `dataset_summary_by_organ.csv` | Per-structure scored count, mean Dice/IoU, accept rate (G1) | Table 1 (MRI) | dev | 50 | per-structure accept rate (spleen 0.50) |
| `primary_model_summary.csv` | Reliability-score AUC + 95% subject-bootstrap CI + both-class PR **+ organ-rate baseline** (G7/G2) | Primary (MRI) / Table 2 | dev | 3 | **pred-IoU RF 0.919 [0.890, 0.947]** vs **organ-rate baseline 0.815 [0.766, 0.861] — CIs disjoint**; PR-AUC(accept, minority) 0.782 |
| `calibration_metrics.csv` (+`_no_intensity`) | ECE/MCE/Brier, none vs **isotonic vs sigmoid/Platt**, nested grouped (G6) | Calibration (MRI) | dev | 3 | **none ECE 0.036; isotonic 0.054, Platt 0.061 — both WORSE** → use uncalibrated (see G6 caveat) |
| `iou_regression_oof.csv` | OOF predicted IoU | Regression (MRI) | dev (in-sample) | 548 | RF Spearman 0.807, R² 0.604; regress@0.90 AUC 0.919 |
| `threshold_sweep_regression.csv` (+`_no_intensity`) | One-model binary metrics at each cutoff | Threshold (MRI) | dev | 5 | AUC rises 0.887(0.70)→0.978(0.95) |
| `conformal_summary.csv` (+`_no_intensity`) | Coverage/width/Spearman, all variants @80/90 | Conformal (MRI) / Table 5 | dev | 10 | **uncal 45.4% → CQR 78.6%@80 (medW 0.158), 87.0%@90 (0.235)** |
| `conformal_by_family.csv` (+`_no_intensity`) | Per-family coverage + cal sizes (feasibility floor) | Conformal supp. | dev | 12 | Mondrian noisy at this scale — report, don't recommend |
| `conformal_intervals_oof.csv` | Per-mask bounds | Conformal (MRI) | dev | 548 | — |
| `volume_label_oof.csv` (+`_no_intensity`) | Per-mask vol_err/vol_bad/iou/features | Volume label (MRI) / Table 6 | dev | 516 | AUC 0.839, PR(bad) 0.537, ECE 0.038; **silent-failure raw 52.5% ⚠in-sample, conditional 69.5%** |
| `volume_label_ablation.csv` (+`_no_intensity`) | Volume-label feature-group ablation | Volume label (MRI) | dev | 12 / 10 | **size becomes load-bearing (drop −0.050); component collapses (−0.006)** — flips vs CT |
| `clinical_decision_flips.csv` | Splenomegaly flips (MRI) | Clinical (MRI) | dev | 14 | **1 flip / 14** — anecdotal (tiny n); not caught by the filter |
| `size_tolerance.csv` | Per-organ tolerance recomputed from MRI spacing | Motivation (MRI) | dev | 50 | voxel tolerances tighter than CT (adrenal 0.12 vox) → explains low accept |

Root files:
- `primary_model_comparison.csv` — 3 rows (CT, MRI+int, MRI−int): accept rate, direct-classifier AUC (balanced & unbalanced), both-class PR-AUC, minority class. (MRI weighting: balanced 0.910 vs unbalanced 0.908 — negligible.)
- `silent_failure_summary.csv` — the "correct-volume-but-low-IoU" number, **like-for-like** across modalities: **raw rate** (of all GT-present masks) and **conditional rate** (of IoU<0.90 masks, the fraction with vol_err<0.10). CT **raw 13.1% / conditional 52.8%**; MRI **raw 52.5% / conditional 69.5% ⚠in-sample**. Compare the *conditional* rate across modalities (the raw rate is dominated by MRI's far higher reject base rate). Both modalities: a volume-only check misses **most** spatially-wrong masks.

---

## Provenance
- CT results: `report_ct_val/` (dataset, classifier, calibration, ablations) + `report_ct_val/regression_phase2/` (regression/conformal/volume/clinical/size, Aug 24).
- MRI results: `report_mri_dev/regression_intensity/` and `.../regression_no_intensity/` (Aug 24, modality driver on the carved dev subset).
- Code: `experiments/pipeline/regression.py` (+ `normalize_columns` schema shim, feasibility-floor Mondrian) and `experiments/pipeline/regression_modality.py`; ablations from `experiments/pipeline/ablations.py`.
- **Consolidation pass** (`experiments/pipeline/paper_consolidate.py`) produced the derived summaries: `ct/primary_model_summary.csv`, `mri/primary_model_summary.csv`, `mri/dataset_combined_metrics.csv`, `mri/dataset_summary_by_organ.csv`, `mri/calibration_metrics.csv` (+`_no_intensity`), `silent_failure_summary.csv`. Re-runnable; G7 bootstraps saved OOF (no retraining), G1 subsets the existing MRI-train table, G6 fits only on the MRI dev subset.
- Copied files were **copied, not moved**. Regeneration touched only CT val / MRI dev; **test splits (CT & MRI) untouched.**
