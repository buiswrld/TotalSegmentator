# SSAT research additions to TotalSegmentator

This fork adds per-mask reliability assessment to TotalSegmentator. Original
TotalSegmentator installation, inference, licensing, and citations are documented
in [README.md](README.md). Its upstream code and license are preserved.

This is the research-code release for **Reference-Free Per-Mask Reliability
Assessment for Whole-Body Anatomical Segmentation Across CT and MRI**, accepted
to the **GenAI4Health Workshop at NeurIPS 2026**. Authors: Aahil Ansari, Soham
Ravindra Lokhande, Ryan Bui, and Kevin Zhu.

**Reproduction status.** The saved primary predictions, CT triage results, and
within-structure results pass the supplied verification checks. Reconstructed
stable fold ordering recovers the saved baselines; forest refits are close but
not bitwise identical. Historical runs use different populations and must not
be combined. See [the audit](docs/ssat/REPOSITORY_AUDIT.md), including its
correction to the earlier CT test-isolation claim. Acceptance of the paper does
not establish clinical validity or guarantee exact numerical refitting.

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

This prints saved and refitted scores. It explicitly uses stable ordering of equally sized subject groups. This matches
all saved CT baseline predictions and the MRI baseline AUC. Forest refits remain
slightly different; see the audit. Other historical scripts retain their original
GroupKFold behavior, which can differ across environments.

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


## Code and data availability

The research code, retained per-mask result tables, and provenance records are
available at https://github.com/buiswrld/TotalSegmentator. The primary paper
results are under `experiments/paper_final/`; reviewer analyses are under
`rebuttal_results/`. `experiments/paper_final_v2/` is a separate full-dataset
analysis and does not replace the main paper protocol.

Original CT/MRI images, full generated masks, and fitted QC model binaries are
not distributed with this research release. Obtain source datasets and
TotalSegmentator weights from their original providers, following their terms
and access requirements. Dataset links and upstream citations are retained in
the [original TotalSegmentator documentation](README.md#totalsegmentator).
For data provenance, the processed MRI training/test tables cover 190/49
subjects, while historical metadata records 561/55 official train/test subjects;
the paper development subset has 38 subjects. These counts describe different
populations.

## Citation and acknowledgments

For the reliability-assessment research, cite:

```bibtex
@misc{ansari2026referencefree,
  title = {Reference-Free Per-Mask Reliability Assessment for Whole-Body Anatomical Segmentation Across CT and MRI},
  author = {Ansari, Aahil and Lokhande, Soham Ravindra and Bui, Ryan and Zhu, Kevin},
  year = {2026},
  note = {Accepted to the GenAI4Health Workshop at NeurIPS 2026},
  howpublished = {\url{https://github.com/buiswrld/TotalSegmentator}}
}
```

This citation records workshop acceptance; no DOI or proceedings identifier is
claimed. Add the public paper URL when available. GitHub citation metadata is
provided in `CITATION.cff`.

This repository is a fork of
[TotalSegmentator by Jakob Wasserthal and collaborators](https://github.com/wasserth/TotalSegmentator).
Our contributions are the reliability descriptors, quality-prediction and
analysis workflows, and associated research results; the segmentation models
and inherited inference implementation are TotalSegmentator's work. Cite the
relevant TotalSegmentator CT/MRI publications, source datasets, and nnU-Net in
addition to this paper when using those components. Original attribution and
the Apache-2.0 LICENSE are retained; weights and datasets retain their own terms.

We thank Algoverse AI Research for research support and Lambda for GPU compute
support, as well as the TotalSegmentator maintainers, public dataset
contributors, and developers of the open-source software used in the study.
The paper authors declare no competing interests.
