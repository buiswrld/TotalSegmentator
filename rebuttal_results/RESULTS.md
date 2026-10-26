# Rebuttal test-set evaluation: results

## Freeze record

- Freeze commit: **e14083a892fea96f8bde436aacb367d29f82c763** (`e14083a`)
- Freeze timestamp: **2026-10-13 05:27:30 -0600**
- Branch: `rebuttal-test-eval` (cut from `73f8779`)
- Config: `rebuttal_results/frozen_config.yaml`
- Audit: `rebuttal_results/CONTAMINATION_AUDIT.md`

Every modelling, feature, calibration, threshold, population, and analysis choice was
fixed in the frozen config and committed above BEFORE any held-out test label (IoU, Dice,
NSD, accept) was read. No choice in this document was tuned after seeing a test result.

## Time-forced deviation from the frozen plan

The rebuttal deadline left only a few hours, so NO new inference was run for these
deliverables. Only outputs already on disk were used. The freeze stands unchanged; this is
a scope reduction, not a change to any modelling choice.

Consequences:
- **MRI prediction regeneration was not performed.** The frozen config's B3 rule therefore
  takes its fallback branch: the existing dev and test tables are used for Requests 1 to 3,
  and the 6 MRI test subjects that produced no predictions in `mr_full_run` (s0078, s0150,
  s0458, s0480, s0523, s0580) remain unscored. MRI test results cover 49 of 55 subjects.
- **CT held-out test inference was not run before these deliverables.** CT test results are
  therefore not reported here. CT Requests 2, 3, and 4 are computed on the CT **dev**
  out-of-fold predictions and are labeled as dev, not test, everywhere.
- **Surface Dice (Request 4) is computed on CT dev only.** MRI NSD needs regenerated
  predicted masks and was not done.
- CT test inference is started on CPU in the background only after Requests 1 to 4 are
  delivered (see the end of this document), for the discussion period / camera-ready.

## Reporting stance

- **CT held-out test is the primary result.** It is a first-time, clean evaluation. No CT
  test prediction, descriptor table, or label existed anywhere before this rebuttal.
- **MRI held-out test is a previously exposed split.** It was scored end to end by a
  teammate on 2026-08-03, about three weeks before the final MRI modelling choices
  (2026-08-24), and it is also in-sample for the segmenter (MRI has no official validation
  split). `CONTAMINATION_AUDIT.md` is cited beside every MRI test number. MRI test is a
  supporting result, not a clean generalization test.

## Implementation facts recorded at freeze (config B1)

1. The paper Methods states 19 reference-free descriptors and 400 trees for both modalities.
   In code the CT classifier trains on 20 raw columns (the constant `is_empty` column is kept
   by `train.py`) while the MRI regressor trains on 19 (the constant column is dropped by
   `regression.py`). The constant column changes CT out-of-fold AUC by less than 0.001
   (20-feature refit 0.8707 vs 19-feature refit 0.8700). Both feature sets are frozen exactly
   as they produced the paper numbers.
2. The MRI primary is a random-forest IoU regressor (300 trees); its predicted IoU is the
   reliability score. The CT primary is a random-forest accept classifier (400 trees). Seed 0
   everywhere; subject-grouped GroupKFold (5 splits) on dev; 1000-draw subject bootstrap for
   95% percentile CIs.

## Phase 0 findings carried forward

- Dev reproduction PASSED to 3 decimals: CT primary 0.8707 (target 0.871), CT organ-rate
  0.7608 (0.761), MRI primary 0.9187 (0.919), MRI organ-rate 0.8150 (0.815).
- MRI `meta.csv` has only train (561) and test (55); no validation split. CT `meta.csv` has
  train (1082), val (57), test (89).
- Six MRI official-test subjects (s0078, s0150, s0458, s0480, s0523, s0580) produced zero
  rows in the earlier `mr_full_run` because upstream inference wrote no predicted masks for
  them. Their images and all 50 reference masks are present and readable. They are
  re-attempted in Phase 2.

---

## Request 1: held-out test-set evaluation

### CT

CT held-out test inference was not run (time-forced deviation above). No CT test number is
reported in this request. CT test inference is queued on CPU after Requests 1 to 4.

### MRI (49 of 55 held-out test subjects)

Caveat on every number below: the MRI test split is PREVIOUSLY EXPOSED and in-sample for the
segmenter. See `rebuttal_results/CONTAMINATION_AUDIT.md`. This is a supporting result, not a
clean generalization test.

The frozen MRI primary (random-forest IoU regressor, 300 trees, 19 features, seed 0) and the
frozen per-structure organ-rate baseline were refit on the 38 dev subjects (548 modelled
masks, dev accept rate 23.0 percent) and scored on the existing 49-subject test table (929
modelled masks, test accept rate 17.1 percent). Subject bootstrap, 1000 draws, 95 percent
percentile CI. Source table: `rebuttal_results/mri/request1_test_metrics.csv`.

| Model | ROC-AUC | PR-AUC (accept) | PR-AUC (reject) |
|---|---|---|---|
| RF IoU regressor (primary) | **0.923** [0.903, 0.941] | 0.752 [0.685, 0.812] | 0.982 [0.973, 0.989] |
| Organ-rate baseline | 0.867 [0.833, 0.903] | 0.551 [0.459, 0.651] | 0.963 [0.943, 0.978] |

Reading: on exposed MRI test the primary regressor reaches ROC-AUC 0.923, essentially equal
to its dev out-of-fold value (0.919), and clears the organ-rate baseline (0.867) with
non-overlapping bootstrap intervals on ROC-AUC and PR-AUC(accept). PR-AUC(accept) is the
honest headline for the minority accept class (17.1 percent prevalence): 0.752 for the model
vs 0.551 for the baseline.

Accept-rate reconciliation: the 17.1 percent used here is 159 accepted masks over the 929
non-empty modelled test masks (159/929); the paper's 16.9 percent is the same 159 accepts
over all 940 test rows, which includes the 11 empty-prediction masks in the denominator
(159/940). Same numerator, different denominator (non-empty modelled vs all rows).

### Model versus organ-rate baseline, dev and test side by side

Paired subject bootstrap (model minus baseline ROC-AUC on the same resampled subjects), 1000
draws, 95 percent percentile. Source: `rebuttal_results/mri/request1_model_vs_baseline_paired.csv`.

| Split | Model AUC | Baseline AUC | Margin (model minus baseline) | 95 percent CI of margin |
|---|---|---|---|---|
| MRI dev | 0.919 | 0.815 | **0.104** | [0.054, 0.160] |
| MRI test (exposed) | 0.923 | 0.867 | **0.056** | [0.027, 0.083] |

Plainly: the model still beats the organ-rate baseline on exposed test (paired margin 0.056,
CI excludes 0), but the margin shrank by nearly half from dev (0.104) to test (0.056). The
baseline itself rose from 0.815 to 0.867, meaning structure identity alone predicts
acceptability better on the test subjects, which compresses the model's added value.

### MRI no-intensity secondary on test (frozen)

The frozen no-intensity variant (14 features, the 5 HU columns dropped) refit on dev and
scored on test gives ROC-AUC **0.924** [0.904, 0.942], PR-AUC(accept) 0.754, PR-AUC(reject)
0.982, indistinguishable from the with-intensity primary (0.923). HU-style intensity features
add nothing on MRI, as expected since MRI intensities are not Hounsfield units. Source:
`rebuttal_results/mri/request1_no_intensity_test.csv`.

### The 6 unscored MRI test subjects (bound)

Source: `rebuttal_results/mri/request1_unscored6_refcounts.csv` and
`request1_accept_rate_bound.csv`. Reading reference-mask contents here is post-freeze
(Phase 3), which is allowed; during Phase 0 only file existence was checked.

All 6 subjects (s0078, s0150, s0458, s0480, s0523, s0580) have **entirely empty reference
segmentations**: 0 of 50 structures has any voxel (total nonzero voxels across all masks is
0), while each subject's `mri.nii.gz` is intact (2.0 to 7.7 million nonzero voxels). This
is the fundamental reason they produced zero rows in `mr_full_run`: there is no ground truth
to score, independent of whether inference ran. Each has **0 reference structures with at
least 20 voxels**, so the worst-case assumption (every eligible reference structure is a
failed empty prediction) adds nothing:

- Accept rate excluding the 6 (49 subjects, 159 of 929 modelled masks): **17.1 percent**.
- Accept rate worst case (6 subjects contribute 0 eligible reference structures): **17.1
  percent**. The bound is tight because these subjects carry no scoreable ground truth.

## Request 2: review-budget triage (10 / 20 / 30 percent)

Rank every modelled mask by model risk (1 minus predicted acceptability), send the riskiest
10, 20, and 30 percent to review, and measure the caught-reject rate (the fraction of all
true rejects that land in the reviewed set). Baseline is the frozen organ-rate ranking.
Random (caught rate equals the budget) and oracle (all rejects first) frame the scale.
Sources: `rebuttal_results/ct/request2_triage_dev_oof.csv`,
`rebuttal_results/mri/request2_triage_test.csv`.

Tie-breaking: the organ-rate baseline gives one score per structure, so masks tie at the
budget boundary. We use fractional (expected) tie-breaking, where masks tied at the cutoff
share the remaining review slots proportionally. This is the fair way to score a tied ranker
and is the expected recall under uniform random tie-breaking; the continuous model scores
have no boundary ties, so for the model this reduces to a hard top-k. Request 2 is a new
rebuttal analysis (the frozen pipeline reports organ-rate only as an OOF AUC and has no
budget-triage code), so the fractional rule is stated here rather than inherited.

### CT (DEV out-of-fold, 400-tree randomforest_[raw]; labeled dev, not test)

| Budget | Model caught | Baseline caught | Random | Oracle |
|---|---|---|---|---|
| 10 percent | **32.8 percent** | 27.3 percent | 10.0 percent | 39.3 percent |
| 20 percent | **56.8 percent** | 45.2 percent | 20.0 percent | 78.4 percent |
| 30 percent | **73.3 percent** | 57.3 percent | 30.0 percent | 100 percent |

CT dev has a 74.5 percent accept rate, so triage has real headroom: reviewing the riskiest
30 percent of masks catches 73 percent of all rejects with the frozen model, versus 57
percent for the organ-rate baseline and 30 percent for a random queue.

### MRI (held-out TEST, 49 subjects, PREVIOUSLY EXPOSED; see CONTAMINATION_AUDIT.md)

The honest headline for MRI is the accept-class separation, not the review-budget recall:
**PR-AUC(accept) 0.752 for the model vs 0.551 for the baseline** (Request 1). The
review-budget recall below is prevalence-limited and should be read with that caveat.

| Budget | Model caught | Baseline caught | Random | Oracle |
|---|---|---|---|---|
| 10 percent | 12.1 percent | 11.9 percent | 10.0 percent | 12.1 percent |
| 20 percent | 24.0 percent | 23.8 percent | 20.0 percent | 24.2 percent |
| 30 percent | 36.1 percent | 35.7 percent | 30.0 percent | 36.2 percent |

The MRI test reject rate is 82.9 percent (accept rate 17.1 percent), so a small review
budget is prevalence-limited: even a perfect oracle catches only 12 percent of rejects at a
10 percent budget, because there are far more rejects than review slots. The model sits
essentially on the oracle line (precision in the reviewed budget is 1.00, 0.99, 0.996 at 10,
20, 30 percent), meaning almost every mask it flags is a true reject. The organ-rate
baseline looks nearly as good on this recall metric only because the pool is overwhelmingly
rejects. The model advantage that ROC-AUC (0.923 vs 0.867) and PR-AUC(accept) (0.752 vs
0.551) show does not translate into triage headroom on a pool this saturated with rejects.
This reflects the MRI segmenter's low accept rate (in-sample, secondary modality), not a
failure of the reliability model.

### Auto-accept (POST HOC, not pre-specified; MRI test, exposed)

The complementary and more informative direction on a reject-heavy pool: auto-accept the top
fraction by predicted quality and measure the failure rate (share of true rejects) among the
auto-accepted. Fractional tie-breaking as above. Source:
`rebuttal_results/mri/request2_autoaccept_posthoc_test.csv`.

| Auto-accept | Model failure rate | Baseline failure rate |
|---|---|---|
| Top 10 percent | **17.2 percent** | 36.7 percent |
| Top 20 percent | **39.8 percent** | 46.2 percent |

Auto-accepting the model's top 10 percent yields a pile that is 17.2 percent contaminated,
less than half the baseline's 36.7 percent. The model carves out a cleaner auto-accept set
than structure identity alone, which is the clinically useful statement for MRI even though
absolute failure rates stay high because the MRI segmenter is weak. This analysis is POST HOC
and was not pre-specified in the frozen config.

## Request 3: absolute within-structure performance

Per structure: the accept rate and the model's within-structure ROC-AUC (computed inside
each structure, which removes the easy cross-structure ranking). Eligibility is the frozen
`ablations.within_organ_auc` rule: at least 25 masks and at least 5 of the rarer class.
Sources: `rebuttal_results/ct/request3_within_structure_dev_oof.csv`,
`rebuttal_results/mri/request3_within_structure_test.csv`.

### CT (DEV out-of-fold, 400-tree randomforest_[raw]; labeled dev, not test)

This uses `primary_model_oof.csv`, not the paper's `within_organ_auc.csv`, which came from a
200-tree ablation run.

- Structures: 117 total, 66 eligible for a within-structure AUC.
- Pooled AUC 0.871; **mean within-structure AUC 0.761**, **median 0.773, IQR [0.662,
  0.903]**; pooled-minus-within gap 0.109. The gap says roughly a tenth of the pooled
  performance is cross-structure difficulty rather than within-structure mask ranking, which
  is expected and modest.
- Accept rate across structures: min 0.00, median 0.79, max 1.00.
- Best within-structure AUC: rib_right_7 0.997, iliopsoas_right 0.987,
  lung_upper_lobe_right 0.968, rib_right_9 0.968, lung_lower_lobe_left 0.960.
- **6 of 66 eligible structures fall below chance (AUC below 0.5)**, listed in full:

| Structure | n | Accept rate | Within-structure AUC | Family |
|---|---|---|---|---|
| duodenum | 35 | 0.17 | 0.328 | digestive |
| brachiocephalic_trunk | 37 | 0.57 | 0.423 | great_vessel |
| gallbladder | 30 | 0.40 | 0.444 | solid_abdominal |
| vertebrae_T1 | 37 | 0.86 | 0.456 | vertebrae |
| superior_vena_cava | 39 | 0.82 | 0.473 | great_vessel |
| common_carotid_artery_left | 38 | 0.66 | 0.483 | great_vessel |

  These are small, variable, or low-contrast structures (three of the six are great
  vessels) where the reference-free descriptors carry little or inverted signal about mask
  quality. They are honest weak spots of the approach and worth naming in the paper.

### MRI (held-out TEST, 49 subjects, PREVIOUSLY EXPOSED; see CONTAMINATION_AUDIT.md)

- Structures: 50 total, **only 2 eligible** (autochthon_left, autochthon_right) under the
  same rule. The exposed MRI test is small (929 masks) and its per-structure accept rate is
  very low (median 0.063), so almost no structure reaches 5 accepted masks.
- For the 2 eligible structures: within-structure AUC 0.863 (autochthon_left) and 0.857
  (autochthon_right); pooled AUC 0.923.
- Honest limitation: a within-structure analysis is not meaningfully available for MRI on
  this test. The full per-structure accept-rate table is still written to the CSV for all 50
  structures, but discrimination within a structure cannot be estimated for 48 of them here.

## Request 4: surface Dice (NSD at 1, 2, 3 mm)

### Dice summary (all three splits, from existing tables)

Non-empty modelled masks only. Source: `rebuttal_results/{ct,mri}/request4_dice_summary.csv`.

| Split | Masks | Subjects | Dice mean | Dice median | Dice p05 | Dice p95 | Frac Dice >= 0.90 |
|---|---|---|---|---|---|---|---|
| CT dev | 4017 | 57 | 0.930 | 0.972 | 0.732 | 0.993 | 0.876 |
| MRI dev | 548 | 38 | 0.842 | 0.910 | 0.000 | 0.972 | 0.551 |
| MRI test (exposed) | 929 | 49 | 0.745 | 0.869 | 0.000 | 0.970 | 0.394 |

The CT segmenter is strong (median Dice 0.972), the MRI segmenter weaker and much more
variable (test median 0.869 but mean 0.745 with a heavy low tail, p05 = 0.000), consistent
with the low MRI accept rate seen throughout.

### Normalized Surface Dice (NSD), CT dev only

DEVIATION, documented: the frozen config named DeepMind surface-distance or MONAI
`compute_surface_dice`. Neither is installed, and the time-forced plan forbids environment
changes, so NSD is computed directly with scipy using the standard symmetric surface-overlap
definition (Nikolov 2021): the fraction of each mask's surface voxels lying within the
tolerance of the other mask's surface, averaged over both directions. Surface equals mask
minus its binary erosion; distances use `distance_transform_edt` with millimetre sampling on
the union bounding box plus a 5-voxel margin (exact for tolerances up to 3 mm at 1.5 mm
spacing). The only difference from the named libraries is a voxel surface rather than a
marching-cubes surface. Predicted and reference masks share an identical 1.5 mm isotropic
grid, so no resampling is involved.

Boundary-voxel connectivity: scipy `binary_erosion`'s default structuring element,
`generate_binary_structure(3, 1)`, which is 6-connectivity (face-connected, 6 neighbours). A
surface voxel is a mask voxel with at least one face-neighbour outside the mask.

ANALYTIC VALIDATION (no installs), next to the substitution above. Two concentric solid
spheres, radii 20 mm and 22 mm, on a 1.5 mm isotropic grid, so their surfaces are nominally
2 mm apart. Run through the identical `surf()`/`nsd()` code (`experiments/rebuttal/_nsd.py`,
`request4c_sphere_validation.py`):

- NSD @ 1 mm = **0.0000** (expected near 0: the 2 mm gap exceeds a 1 mm tolerance).
- NSD @ 2 mm = 0.3943 (borderline, as expected when the gap equals the tolerance).
- NSD @ 3 mm = **1.0000** (expected near 1: the 2 mm gap is inside a 3 mm tolerance).
- Measured inner-to-outer surface distance: mean 1.88 mm, median 2.12 mm, max 3.00 mm. This
  equals the geometric 2 mm gap, which confirms distances are in millimetres derived from the
  voxel spacing (a voxel-unit computation would have read about 1.33). 
- Identical-mask control (outer vs outer): NSD = 1.0000 at all three tolerances.

The implementation behaves exactly as the NSD definition requires.

Computed over all 4017 CT dev modelled masks (3979 both-present pairs plus 38 pred-only
false positives scored NSD = 0) in 14 minutes. Source:
`rebuttal_results/ct/request4_nsd_*_dev.csv`, per-mask in `request4_nsd_permask_dev.csv`.

Overall distribution:

| Tolerance | Mean | Median | Std | p05 | p25 | p75 | p95 | Frac >= 0.95 |
|---|---|---|---|---|---|---|---|---|
| 1 mm | 0.805 | 0.860 | 0.182 | 0.430 | 0.757 | 0.928 | 0.961 | 0.117 |
| 2 mm | 0.944 | 0.992 | 0.153 | 0.720 | 0.965 | 0.999 | 1.000 | 0.805 |
| 3 mm | 0.959 | 0.999 | 0.144 | 0.794 | 0.987 | 1.000 | 1.000 | 0.884 |

Surface agreement is high at clinical tolerances (median NSD 0.992 at 2 mm, 0.999 at 3 mm).

Highlighted structures (mean NSD@2mm). Worst are small or deformable: kidney_cyst (n=1,2,
NSD 0.00), prostate 0.67, duodenum 0.81, brain 0.81, urinary_bladder 0.83, portal/splenic
vein 0.84, small_bowel 0.84. Cross-referencing the Request 3 below-chance structures: duodenum
(0.81) and gallbladder (0.87) are genuinely low on surface agreement too, but
brachiocephalic_trunk (0.97), vertebrae_T1 (0.97), superior_vena_cava (0.98) and
common_carotid_artery_left (0.98) have HIGH surface agreement despite below-chance
within-structure AUC. For those great vessels the masks are surface-good and uniform, so there
is little quality variation for the model to rank, not a surface failure.

IoU-vs-NSD confusion (accept = IoU >= 0.90 vs surface-accept = NSD >= 0.95):

| Tolerance | IoU-acc / NSD-acc | IoU-acc / NSD-rej | IoU-rej / NSD-acc | IoU-rej / NSD-rej | Agreement |
|---|---|---|---|---|---|
| 1 mm | 471 | 2522 | 0 | 1024 | 0.372 |
| 2 mm | 2849 | 144 | 383 | 641 | **0.869** |
| 3 mm | 2965 | 28 | 586 | 438 | 0.847 |

NSD >= 0.95 at 2 mm agrees with the IoU >= 0.90 label on 86.9 percent of masks, the best of
the three. At 1 mm the surface cut is far stricter than IoU (no mask is surface-perfect yet
IoU-bad, so the off-diagonal is one-sided); at 3 mm the tolerance starts accepting surfaces
that IoU still rejects (586 masks).

Disagreement (IoU-accept not equal to NSD@2mm-accept) by reference-volume quartile:

| Volume quartile | n | Median ref volume (mm3) | Mean NSD@2mm | Disagreement rate |
|---|---|---|---|---|
| Q1 smallest | 995 | 5,835 | 0.936 | **0.242** |
| Q2 | 995 | 18,205 | 0.947 | 0.079 |
| Q3 | 994 | 42,010 | 0.966 | 0.064 |
| Q4 largest | 995 | 297,469 | 0.962 | 0.144 |

Disagreement is U-shaped and highest for the smallest structures (24 percent): on small
masks a 2 mm surface tolerance and a 0.90 IoU cut diverge most, because a couple of voxels of
boundary error move IoU far more than NSD. The largest structures show a secondary rise (14
percent): their surfaces are almost always within 2 mm (high NSD), yet volume offsets can
still drop IoU below 0.90.

OOF score vs NSD (frozen 400-tree dev out-of-fold randomforest_[raw]):

| Tolerance | Spearman(model, NSD) | Spearman(IoU, NSD) | Model AUC for NSD >= 0.95 | Mean NSD accepted | Mean NSD rejected |
|---|---|---|---|---|---|
| 1 mm | 0.419 | 0.746 | 0.664 | 0.872 | 0.611 |
| 2 mm | 0.355 | 0.724 | 0.751 | 0.989 | 0.812 |
| 3 mm | 0.280 | 0.610 | 0.778 | 0.996 | 0.853 |

The frozen reliability model, trained only on the IoU label, still ranks surface quality: its
score predicts NSD >= 0.95 with AUC 0.66 to 0.78 and is positively correlated with NSD
(Spearman 0.28 to 0.42). IoU itself correlates with NSD more strongly (0.61 to 0.75), as
expected since both measure overlap directly. The accept label separates cleanly on NSD at
every tolerance (at 2 mm, accepted masks average 0.989 vs 0.812 for rejected). The reliability
signal therefore generalizes beyond the single IoU cutoff to a surface-distance metric.

MRI NSD is not computed: it requires the regenerated MRI predicted masks, which the
time-forced plan did not produce (the `mr_full_run` predicted masks were never committed).

## Request 5: CNN baseline (go / no-go)

**Not attempted.** This was always a go/no-go item, and the frozen config recorded it as
"decision pending feasibility review; not committed to run." Under the time-forced deviation
(no new inference, a few hours to the deadline), it is a no-go for this submission:

- A fair image-based CNN baseline needs the raw CT/MRI volumes, a training pipeline, a GPU
  (which is not usable in this environment without an environment change we have ruled out),
  and subject-grouped train/val/test splits that match the frozen protocol. None of that is
  achievable in the time available.
- A baseline trained and compared hastily would be a weak straw man and would misrepresent
  the comparison rather than inform it, which the ground rules forbid.

Recommended framing for the response: defer the CNN baseline to the camera-ready / discussion
period, run under the same frozen splits and the GPU inference environment prepared in the
Option 1 plan, so it is an honest comparison rather than a rushed one.
