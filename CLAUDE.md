# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TotalSegmentator is a Python tool/library (nnU-Net based) for segmenting anatomical structures
(100s of classes across dozens of "tasks") in CT and MR images. It ships as a pip package with a
main CLI (`TotalSegmentator`), several helper CLIs, and a Python API
(`totalsegmentator.python_api.totalsegmentator`).

## Setup

```bash
pip install -e .
pip install pytest Cython fury xgboost
pip install torch==2.6.0 -f https://download.pytorch.org/whl/cpu   # or a CUDA build
```

Model weights are downloaded on first use per-task into `~/.totalsegmentator/nnunet/results`
(override with `TOTALSEG_HOME_DIR` / `TOTALSEG_WEIGHTS_PATH` env vars). Licensed tasks need
`totalseg_set_license -l <key>` (free academic licenses available).

The spine report tests need system binaries: `apt-get install wkhtmltopdf xvfb` (see
`.github/workflows/run_tests.yml`).

## Common commands

```bash
# Lint (ruff via pre-commit; also runs pyupgrade + codespell)
pre-commit run --all-files

# Fast tests that need no GPU/model download
pytest -v tests/test_device_type.py
pytest -v tests/test_config.py
pytest -v tests/test_registry.py
pytest -v tests/test_research_utils.py

# A single test
pytest -v tests/test_registry.py::test_name

# Full end-to-end suite (downloads models, runs real segmentations on CPU, ~slow)
./tests/tests.sh <license_key>

# End-to-end tests only, without re-running the CLI commands in tests.sh
pytest -v tests/test_end_to_end.py

# Spine report tests (needs wkhtmltopdf/xvfb installed)
pytest -v tests/test_spine_report.py

# Sanity-check the CLI/task registry without a GPU or model download
totalseg_info --list-tasks
totalseg_info --classes -ta total --json
TotalSegmentator --list-tasks
```

CI mirrors this: `.github/workflows/lint.yml` (pre-commit), `run_tests.yml` (`tests/tests.sh` +
spine report), `run_tests_os.yml` (`tests/tests_os.py`, cross-OS Python API checks),
`run_tests_subtasks.yml` (`tests/tests_subtasks.sh`, one run per licensed/extra task),
`run_tests_nnunet.yml` (`tests/tests_nnunet.py`).

## Architecture

**Task registry (`registry.py`) is the single source of truth for *what* tasks exist.** It reads
pure data from `map_to_binary.py` (`class_map`, `commercial_models`) — no torch import — so
`totalseg_info` and `--list-tasks`/`--list-classes` can answer instantly without a GPU or model
download. `TASKS` in `registry.py` is the canonical task list; `bin/TotalSegmentator.py` imports
it for its `--task` choices so the CLI and registry never drift apart. Adding a new task means
updating `TASKS`/`commercial_models`/`class_map`, *and* wiring the actual nnU-Net params into the
big if/elif chain described next.

**`python_api.py::totalsegmentator()` is the real orchestration entry point** that both the CLI
and the Python API funnel through. Its core is a long `if task == "...":` chain that maps each
task name to its nnU-Net run parameters: `task_id` (integer or list — a list means an ensemble of
folds/models), `resample` (target spacing; some tasks use a fixed mm isotropic value, others an
explicit `[x,y,z]` — note plans.json stores `[z,y,x]`, reversed when copying values in), `trainer`
class name, `crop` (list of class names from another task to crop the image to before running, or
`None`), `crop_addon` (mm padding around the crop bbox), `model` (`3d_fullres` / `3d_fullres_high`
/ `3d_lowres_high`), and `folds`. Licensed ("commercial") tasks call `show_license_info()` which
exits if no valid license is configured.

Before the main prediction, the function may run one or two cheap low-res auxiliary
segmentations: a rough "total"/"total_mr" pass (3mm or 6mm) to build a `crop`/`cascade` mask when
`crop`, `roi_subset`, or `cascade` is set, and/or a rough body segmentation when `body_seg=True`.
These reuse `nnUNet_predict_image` from `nnunet.py` with `multilabel_image=True` and are then
converted into a binary crop mask via `class_map_inv` lookups.

**`nnunet.py::nnUNet_predict_image`** is the actual nnU-Net inference wrapper (resampling,
cropping, TTA, multi-fold ensembling, writing per-class or multilabel NIfTI output, run-report
plumbing). `python_api.py` never calls nnU-Net directly — everything routes through this
function.

**Statistics and postprocessing** (`statistics.py`, `postprocessing.py`) run after the main
prediction when `--statistics`/`--radiomics`/`--remove_small_blobs` etc. are requested, operating
on the already-produced segmentation.

**`config.py`** owns the `~/.totalsegmentator/config.json` file (anonymous usage stats opt-out,
license number, prediction counter) and the `TOTALSEG_HOME_DIR`/`TOTALSEG_WEIGHTS_PATH`
environment variable resolution used by every other module.

**`bin/`** holds one thin argparse wrapper per console-script entry point (see `setup.py`
`entry_points`); each just parses args and calls into the corresponding library module — put real
logic in the library, not in `bin/`.

**`spine_report/`** is a self-contained sub-package (own logger, HTML/PDF rendering via
wkhtmltopdf, vertebra height measurement) driven by `totalseg_spine_report`.

## Notes for automation / agents

`AGENTS.md` in the repo root documents how to drive this tool non-interactively (discovering
tasks/classes via `totalseg_info --json`, running with `--report` for a machine-readable run
manifest, exit codes, `--statistics`/`--statistics_extra` JSON output). Read it before scripting
against the CLI or Python API — it also documents the `--report` JSON schema in detail.

Do not hardcode task or class names from source in scripts/tests where avoidable — the registry
functions (`totalsegmentator.registry.list_tasks`, `get_task_classes`, `task_registry`) are
generated from the same data the CLI validates against.

## Style

Ruff line-length is 550 (effectively unbounded) — this codebase does not wrap lines. `F401`
(unused import), `F821` (undefined name) and `F841` (unused variable) are ignored, and
`fixable = ["ALL"]` in `pyproject.toml`. Don't fight this with tighter local lint suppressions.
