# SSAT research additions to TotalSegmentator

This fork adds per-mask reliability assessment to TotalSegmentator. Original
TotalSegmentator installation, inference, licensing, and citations are documented
in [README.md](README.md). Its upstream code and license are preserved.

**Release status: review candidate.** Saved paper/rebuttal tables pass the checks
below. Exact refits differ in the available environment. The cross-branch history
also contradicts the October claim that CT test had never been evaluated. Read
[the audit](docs/ssat/REPOSITORY_AUDIT.md) before describing test isolation or using
full-dataset results in the manuscript. The camera-ready paper is not in this
export, so this candidate has not been matched line by line to that PDF.

## Which results belong to which protocol?

| Location | Population and purpose |
|---|---|
| `experiments/paper_final/ct/` | Paper CT development evaluation: 57 validation subjects; 4,017 nonempty modelled masks. |
| `experiments/paper_final/mri/` | Paper MRI development evaluation: 38 subjects carved from the processed MRI training subset; 548 nonempty masks; in-sample for the segmenter. |
| `rebuttal_results/` | Additional CT development triage/within-structure/surface analyses, plus exposed MRI test analyses covering 49 of 55 subjects. Original freeze and audit are historical records; see the correction in the repository audit. |
| `experiments/paper_final_v2/` | Separate August full-dataset analysis. Different populations and fitting protocol; does not replace the paper tables. Some analyses perform grouped CV within an official test population, so folder names do not establish untouched holdout performance. |
| `experiments/eval_runs/` | Committed input tables, manifests, and older results supporting the analyses above. Predictions and model binaries were excluded from the export. |

The names GRAM and MaskGuard appear in historical records for this research line.
They are retained to preserve provenance.

## Install and check saved evidence

Use Python 3.10 or newer in a separate environment. From the repository root:

```bash
python -m pip install -r requirements-ssat.txt
python experiments/verify_saved_results.py
python -m unittest discover -s tests -p test_ssat_release.py -v
python -m pytest -q tests/test_mask_metrics.py
```

The first two checks require no model weights, raw images, GPU, or inference.
The mask-metrics tests additionally require NiBabel and pytest. Dependency ranges
are provided for installation, **not** as a recovered historical lockfile.

To attempt the original development refits:

```bash
python experiments/rebuttal/reproduce_dev.py
```

This prints saved and refitted scores. It is a diagnostic and currently does not
reproduce all historical values in the review environment; see the audit.

## Code entry points

| Entry point | Role |
|---|---|
| `totalsegmentator/mask_metrics.py` | Predicted-mask geometry, connectivity, boundary, and image-intensity descriptors. Intensity features also require the corresponding image. |
| `resources/run_mask_metrics.py` | Descriptor CLI. |
| `experiments/pipeline/` | Six-stage inference/metrics/reference/curation/classification workflow, plus regression and ablations. |
| `experiments/pipeline/ablations.py` | Paper's original short-column-schema ablations, imported by regression and rebuttal code. |
| `experiments/pipeline/ablations_staged.py` | Later CLI ablations for stage-4 curated tables; includes family threshold sensitivity. Kept separately because its API/schema differs. |
| `experiments/rebuttal/` | Reviewer analyses. Use `python -m experiments.rebuttal.<script>` from the root for package imports. |
| `experiments/legacy/combined_mri.py` | Superseded early MRI harness from `sugoiiia-patch-1`; retained for provenance, not the release entry point. |

The generic staged classifier workflow is documented in
[experiments/pipeline/README.md](experiments/pipeline/README.md). Running it is a
new analysis, not an automatic reproduction of every original paper result.
Inference additionally requires the inherited TotalSegmentator installation,
weights and original source datasets. There is no committed deployable QC model.

For continuous IoU prediction on the committed development inputs:

```bash
python experiments/pipeline/regression_modality.py
```

It preserves the CT-validation/MRI-train-dev protocol and writes to
`ssat_outputs/`. It fits the original model families; it is more expensive than
saved-table verification. `regression.py`, `ablations.py`, and
`paper_consolidate.py` also have portable default inputs and separate output paths.

To regenerate the table-only rebuttal analyses into `ssat_outputs/rebuttal/`:

```bash
python -m experiments.rebuttal.request2_triage
python -m experiments.rebuttal.request3_within_structure
python -m experiments.rebuttal.request4_dice_summary
python -m experiments.rebuttal.request4b_nsd_report
```

The first two refit the MRI regressor; their MRI results can differ across
environments. The surface report uses the saved per-mask table. Recomputing that
table from masks requires `SSAT_CT_DATASET` and `SSAT_CT_PREDICTIONS` environment
variables and `python -m experiments.rebuttal.request4_nsd`. Its implementation is
a **boundary-voxel-count approximation**, not the area-weighted DeepMind surface
Dice implementation specified in the original frozen plan. Do not present them
as interchangeable. The historical CT-only inference script requires
`SSAT_CT_DATASET` and `SSAT_CT_TEST_PREDICTIONS`; the MRI subject-file audit script
requires `SSAT_MRI_DATASET`.

## Operational notes

- Named subject splits now require valid metadata and cannot silently fall back
  to all subjects. Reference-table construction rejects `test`, pooled `all`,
  and empty split requests; its default is `train`.
- Multi-process metric calculation and sharded inference come from the later
  GPU branch. Default metric calculation remains single-process. The sharded
  shell helper now propagates failure from any shard.
- Existing inference resume logic skips a subject when any prediction file is
  present. Validate prediction completeness before reusing an interrupted run.
- Historical runbooks are retained under `docs/ssat/HISTORICAL_*`; their machine
  paths, branch names, checkpoint/push commands, and scientific claims are not
  release instructions. The automatic git-commit/push helper remains in branch
  history rather than the candidate working tree.
- No new dataset inference, model-weight download, remote merge, or push was
  performed to prepare this candidate.
