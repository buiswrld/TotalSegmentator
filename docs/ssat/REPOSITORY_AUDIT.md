# SSAT repository consolidation audit

Prepared from `SSAT_repo_review.zip` on 2026-10-26. Default branch in the export
is **master**, not main. Its tip is `8c01e33`; the sole exported working checkout
is clean and contains no additional untracked files. Candidate branch:
`ssat-release-candidate`. No remote repository was modified.

## Baseline and scope

`7e6214a` is the inherited TotalSegmentator commit immediately before the first
SSAT commit on master. It is used as the local historical baseline, not a claim
about today's upstream GitHub tip. Inherited inference/package files are not
rewritten. The pre-existing research addition to `totalsegmentator/statistics.py`
(`calculate_mask_volumes`) is retained without further changes. LICENSE and
setup.py remain unchanged. The existing research .gitignore additions are retained
and extended for regenerated outputs. The original README content remains intact below a
short research-navigation notice.

## Branch decisions

| Exported branch | Tip | Decision |
|---|---|---|
| master | `8c01e33` | Preserve ancestry; early MRI harness is superseded. |
| add-gitignore | `1d38f8a` | Already in the paper/rebuttal ancestry. |
| test-tool | `da99030` | Retain mask-volume helper already in ancestry. |
| qc-mask-metrics | `9edb204` | Retain descriptor module, canonical orientation, tests. |
| sugoiiia-patch-1 | `3d917d6` | Retain its later standalone MRI harness under `experiments/legacy/`. |
| ct-eval-harness | `b66cadc` | Already included through the paper branch. |
| reproduce-ct-eval | `94eadec` | Retain staged pipeline; correct navigation and reproducibility documentation. |
| ablations | `432e670` | Retain original API used by paper regression/rebuttal. |
| ablations-pipeline | `f84bd05` | Retain later staged CLI as `ablations_staged.py`; avoid breaking original imports. |
| regressions | `bbcc359` | Retain continuous IoU, uncertainty, volume and clinical illustration code. |
| final | `73f8779` | Retain committed paper tables as original evidence. Branch name alone does not identify the newest work. |
| remote-gpu-inference | `c03cd55` | Selectively integrate multiprocessing/sharding/streaming/split tools, useful fixes, and separately labeled full-data tables. Preserve its research notes/runbook as historical documentation. Do not replace the paper protocol with its test-population CV protocol. |
| rebuttal-test-eval | `a3337f3` | Candidate base: adds October analyses and freeze/audit records to `final`. |

All original exported branch tips are retained as archive references in the
review repository and distributed bundle. Older work is recoverable without
keeping every superseded implementation as an active entry point.

## Finding 1: prior CT test evaluation exists on another branch

The October `rebuttal_results/CONTAMINATION_AUDIT.md` and `RESULTS.md` say no CT
test predictions, descriptor table, labels, or evaluation existed before the
October freeze. That statement is contradicted by the August branch:

- `experiments/eval_runs/ct_official_split/metrics/split_by_official_manifest.json`
  records `--ct-val-into test` and the official train/val/test counts.
- `experiments/eval_runs/ct_official_split/results/test/test_manifest.json`
  records a completed 16-model test evaluation at **2026-08-31T10:56:12 UTC**,
  9,652 curated rows.
- The corresponding raw evaluation table contains **140 subjects**, of which
  **53 overlap the 57 paper CT validation subjects** and **87 lie outside that
  validation set**. The committed split manifest identifies this population as
  official val+test. The original dataset meta.csv is not included, so per-ID
  split assignments cannot independently be rechecked here.
- `experiments/paper_final_v2/` also includes later regression/CV analyses over
  this evaluation population. Grouped CV inside a test population is not a
  dev-fit model evaluated once on an untouched test set.

This proves earlier CT evaluation records existed in the repository history.
It does **not** establish who read those results or whether they influenced the
paper. Do not infer intent or misconduct. It does mean a repository-wide claim
that CT test had never been evaluated is unsupported. Historical freeze/audit
files are preserved, not silently rewritten; this audit supersedes their
repository-wide absence claim. CT validation OOF results remain development
results; this finding does not itself prove subject leakage within their folds.

MRI's October records already disclose prior MRI test exposure. The exact
segmenter training overlap of each MRI population is not independently verified
by this code export; preserve the paper's conservative caveats rather than
making stronger claims from the split name.

## Finding 2: fold tie ordering explains the large refit drift

Saved evidence checks pass:

| Check | Value |
|---|---:|
| CT primary saved OOF AUC | 0.8707038428 |
| CT organ-rate saved OOF AUC | 0.7608265314 |
| MRI primary saved OOF AUC | 0.9186978109 |
| CT primary mean within-structure AUC | 0.7613075541 (66 eligible structures) |
| CT review-budget table | Exact match, including fractional boundary ties |

Fresh five-fold development fits in the review environment yield:

| Model | Historical saved | Current refit |
|---|---:|---:|
| CT RF classifier, 400 trees, 20 raw columns | 0.8707 | 0.8756 |
| CT organ-rate baseline | 0.7608 | 0.7564 |
| MRI RF IoU regressor, 300 trees, 19 columns | 0.9187 | 0.9111 |
| MRI organ-rate baseline | 0.8150 (historical summary) | 0.8345 |

Environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0,
SciPy 1.17.0. Historical exact analysis dependency versions and explicit fold
assignments were not found in the exported manifests. A follow-up diagnostic
identified the major source: stable sorting of equally sized subject groups.
With stable tie ordering, **all 4,017 saved CT organ-rate predictions match
exactly**, and the MRI organ-rate AUC is 0.8150436, matching its historical
summary. Default quicksort in the current GroupKFold environment gives the
different baseline values above. This is direct evidence that fold assignment,
not only forest randomness, caused the large discrepancy.

The new `experiments/pipeline/folds.py` makes that ordering explicit, and
`reproduce_dev.py` uses it. With the frozen 20-column CT/19-column MRI settings,
forest refit AUCs become **0.8704 CT** and **0.9186 MRI**, versus saved 0.8707
and 0.9187. Small residual estimator/preprocessing differences remain unresolved.
Stable tie ordering is a reconstructed rule validated against saved predictions;
it is not an original fold manifest recovered from the authors. Other historical
analysis entry points retain their original GroupKFold imports. Do not advertise
bitwise reproduction of all experiments. The evidence does not justify labeling
the saved results fabricated.

The old `reproduce_dev.py` default also dropped CT's constant `is_empty` column,
despite the frozen configuration keeping it. The candidate corrects that check
to 20 columns; a small residual mismatch remains after stabilizing fold ordering. Saved tables were not altered to match new
fits. New output paths prevent incidental overwrites of original evidence.

## Finding 3: surface analysis differs from frozen specification

The freeze names DeepMind surface-distance or MONAI surface Dice. The actual
October implementation counts eroded boundary voxels and uses physical-distance
transforms. This is a boundary-voxel approximation, not area-weighted surface
Dice. The candidate retains that computation, shares the function with its
sphere diagnostic, and labels the limitation. A small sphere check supports
distance units; it does not establish equivalence to the specified libraries.
No replacement NSD experiment was run.

## Consolidation changes

- Preserve `train.py`, `test.py`, paper ablation API, and development population
  selection; retain later staged ablations under a distinct entry point.
- Preserve the calibration dictionary keys required by the original scripts
  when bringing in the later common helpers.
- Bring in sharded/streamed inference, parallel metrics, and official split
  regrouping. Sort parallel metric rows into subject selection order; correct
  the sharded shell helper to return failure if any child fails.
- Reject silent named-split fallback and pooled/test reference construction.
- Apply the later clinical cutoff-direction and fresh classifier comparison
  fixes. The later volume-ablation schema patch is unnecessary in the retained
  paper API, whose feature groups already use short column names.
- Replace machine-specific paths in active analysis entry points; separate
  regeneration outputs from committed evidence. Historical notes/configs retain
  original paths as provenance.
- Remove the team's tracked generated example segmentation from the candidate
  tree; keep it in original history. Do not remove inherited example assets.
- Keep automatic git checkpoint/push helper in history rather than release entry
  points. No commits were pushed or remote branches deleted.

## Validation and limits

Passed saved-table verification; five stdlib unit checks for split protection,
disjoint pooled windows, compatible calibration schema, fractional review ties,
and physical surface-distance units; Python syntax compilation; shell syntax;
and the staged-ablation CLI help check. Development model refits ran to
completion, with the discrepancies above.

NiBabel and pytest are absent in this environment, and package retrieval failed;
the existing mask-metrics test suite and image-dependent stages could not be
executed. No GPU inference or full historical sweep was run. Predictions, model
binaries, original raw datasets, and the current camera-ready PDF are absent
from this export. This candidate is suitable for review and consolidation, not
an assertion that every manuscript claim and end-to-end experiment is verified.

Before publication: reconcile the CT test-exposure wording, recover the refit
environment/folds if exact reproduction is promised, run mask-metrics tests with
dependencies installed, and compare the actual camera-ready paper to the
retained protocol and surface-metric implementation.
