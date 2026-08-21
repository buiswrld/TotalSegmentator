"""
Shared helpers for the staged QC pipeline (run_inference -> compute_metrics ->
build_reference_table -> curate_dataset -> train -> test).

Every stage is a standalone CLI script under experiments/pipeline/ - none of them
hardcode a dataset path, output path, or dataset-specific constant (image filename,
TotalSegmentator task, etc.); everything is a flag, with MODALITY_DEFAULTS supplying
sensible defaults for --modality ct/mr. This module holds only the logic duplicated
across those scripts: subject discovery from meta.csv, canonical-RAS loading, and
argparse/manifest boilerplate.
"""

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
from sklearn.metrics import brier_score_loss

from totalsegmentator.qc_columns import (
    COL_N_MASKS, COL_BIN_MEAN_PREDICTED, COL_BIN_OBSERVED_RATE, COL_CALIBRATION_GAP,
    COL_BRIER_SCORE, COL_RELIABILITY, COL_RESOLUTION, COL_UNCERTAINTY,
)

# Per-modality defaults for the dataset layout / TotalSegmentator invocation. Override
# any of these per-run with --image-name / --task / --device on the stage CLIs.
MODALITY_DEFAULTS = {
    "ct": {"image_name": "ct.nii.gz", "task": "total"},
    "mr": {"image_name": "mri.nii.gz", "task": "total_mr"},
}


def resolve_modality(args):
    """(image_name, task) for args.modality, with --image-name/--task overrides applied."""
    defaults = MODALITY_DEFAULTS[args.modality]
    image_name = args.image_name or defaults["image_name"]
    task = args.task or defaults["task"]
    return image_name, task


# ---------------------------------------------------------------- subject discovery

def find_columns(header):
    """(id_column, split_column) guessed from a meta.csv header row, or (None, None)."""
    lower = [h.lower() for h in header]
    id_col = next((header[lower.index(c)] for c in
                   ("image_id", "subject", "id", "image", "name") if c in lower), None)
    split_col = next((header[lower.index(c)] for c in
                      ("split", "partition", "subset", "set", "fold") if c in lower), None)
    return id_col, split_col


# Fixed seed for the "all" pooled mode's deterministic shuffle (see get_subjects) - keeps
# repeated invocations reproducible and matches this codebase's existing --seed 0 default
# used elsewhere (train.py, ablations.py).
POOLED_SHUFFLE_SEED = 0


def get_subjects(dataset_dir, split, limit, offset=0):
    """Subject ids from dataset_dir/meta.csv filtered by split; falls back to all sXXXX folders.

    split="all" bypasses the split-column filter entirely and pools every subject in
    meta.csv regardless of its original train/val/test label, in a deterministic
    shuffled order (fixed seed, see POOLED_SHUFFLE_SEED) - used to build a QC-pipeline-
    specific train/test partition independent of the split TotalSegmentator itself was
    trained/evaluated on. offset lets a caller carve out a disjoint window of this
    pooled, shuffled list (e.g. classifier-train = subjects[0:493], classifier-test =
    subjects[493:493+123]).
    """
    meta = os.path.join(dataset_dir, "meta.csv")
    subjects = []
    pooled = split == "all"
    if split and os.path.exists(meta):
        with open(meta, newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096); fh.seek(0)
            delim = ";" if sample.count(";") > sample.count(",") else ","
            reader = csv.DictReader(fh, delimiter=delim)
            id_col, split_col = find_columns(reader.fieldnames or [])
            if id_col and (pooled or split_col):
                subjects = [row[id_col].strip() for row in reader
                            if pooled or row.get(split_col, "").strip().lower() == split.lower()]
            else:
                print("  meta.csv columns not recognized; using all subject folders")
    if not subjects:
        subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dataset_dir, "s0*"))
                          if os.path.isdir(p))
    if pooled:
        subjects = sorted(subjects)
        random.Random(POOLED_SHUFFLE_SEED).shuffle(subjects)
    return subjects[offset:offset + limit] if limit else subjects[offset:]


# ---------------------------------------------------------------- image loading

def load_canonical(path):
    """Load a NIfTI reoriented to closest canonical RAS. Returns (img, original_axcodes)."""
    img = nib.load(path)
    return nib.as_closest_canonical(img), tuple(nib.aff2axcodes(img.affine))


# ---------------------------------------------------------------- inference

def ensure_predictions(dataset_dir, predictions_dir, subject, image_name, task,
                       device, extra_args, run_ts=True):
    """Return the prediction dir for a subject, running TotalSegmentator if needed.

    Skips subjects that already have predictions on disk (the resumability mechanism
    every stage relies on) - re-running stage 1 after a partial/interrupted run only
    does the missing work.
    """
    pred_dir = os.path.join(predictions_dir, subject)
    if glob.glob(os.path.join(pred_dir, "*.nii.gz")):
        return pred_dir
    if not run_ts:
        return None
    image_path = os.path.join(dataset_dir, subject, image_name)
    if not os.path.exists(image_path):
        return None
    os.makedirs(pred_dir, exist_ok=True)
    print(f"    running TotalSegmentator on {subject} ...")
    cmd = ["TotalSegmentator", "-i", image_path, "-o", pred_dir, "-ta", task]
    if device:
        cmd += ["-d", device]
    cmd += extra_args
    subprocess.run(cmd, check=True)
    return pred_dir


# ---------------------------------------------------------------- calibration metrics
# (used by stage 6, test.py - AUC says the model RANKS masks correctly; these say
# whether its scores are usable as probabilities, e.g. a score of 0.8 meaning the mask
# really is good about 80% of the time.)

def ece_mce(y, p, n_bins=10, equal_count=True):
    """Expected and maximum calibration error.

    Equal-count bins by default: with equal-WIDTH bins most of the mass lands in one or
    two bins when scores are skewed, so the average would be dominated by nearly-empty
    bins. Equal-count bins give every bin the same weight, the more honest summary here.
    """
    n = len(y)
    if equal_count:
        order = np.argsort(p)
        edges = [order[i] for i in np.linspace(0, n, n_bins + 1)[1:-1].astype(int)]
        bins = np.split(order, [np.searchsorted(np.sort(p), p[e], "left") for e in edges])
        bins = [b for b in bins if len(b)]
    else:
        idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
        bins = [np.where(idx == b)[0] for b in range(n_bins)]
        bins = [b for b in bins if len(b)]
    ece = 0.0
    mce = 0.0
    rows = []
    for b in bins:
        conf, acc = p[b].mean(), y[b].mean()
        gap = abs(acc - conf)
        ece += len(b) / n * gap
        mce = max(mce, gap)
        rows.append({COL_N_MASKS: len(b), COL_BIN_MEAN_PREDICTED: conf,
                     COL_BIN_OBSERVED_RATE: acc, COL_CALIBRATION_GAP: acc - conf})
    return ece, mce, rows


def brier_decomposition(y, p, n_bins=10):
    """Brier = reliability - resolution + uncertainty (lower Brier is better).

    reliability: how far predictions sit from observed rates (0 is perfect calibration)
    resolution:  how much predictions vary from the base rate (higher = more informative)
    uncertainty: the base rate's own variance - a property of the data, not the model
    """
    n = len(y)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    base = y.mean()
    rel = res = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        nk, pk, ok = sel.sum(), p[sel].mean(), y[sel].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - base) ** 2
    return {COL_BRIER_SCORE: brier_score_loss(y, p), COL_RELIABILITY: rel / n,
            COL_RESOLUTION: res / n, COL_UNCERTAINTY: base * (1 - base)}


# ---------------------------------------------------------------- CLI / manifest helpers

def add_dataset_args(parser: argparse.ArgumentParser, require_dataset_dir=True):
    parser.add_argument("--dataset-dir", required=require_dataset_dir,
                        help="Root dataset directory (has meta.csv and one subfolder per subject).")
    parser.add_argument("--modality", choices=list(MODALITY_DEFAULTS), default="ct",
                        help="Selects default --image-name/--task (default: ct).")
    parser.add_argument("--image-name", default=None,
                        help="Override the per-modality default image filename (e.g. ct.nii.gz).")
    parser.add_argument("--task", default=None,
                        help="Override the per-modality default TotalSegmentator task (e.g. total).")
    parser.add_argument("--split", default="test",
                        help="meta.csv split to use; omit/empty to use every subject folder. "
                             "'all' pools every subject regardless of split label, deterministically "
                             "shuffled (see get_subjects/POOLED_SHUFFLE_SEED in common.py).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of subjects processed. Default: no cap.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many subjects before applying --limit (e.g. to carve a "
                             "disjoint window out of --split all's pooled, shuffled subject list).")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(path, **fields):
    """Write a small JSON manifest recording how an artifact was produced: CLI args,
    git commit, timestamp, plus whatever stage-specific fields are passed in. This is
    what lets a later stage (or a person, months later) reconstruct exactly how a CSV
    or model file was generated without re-reading the run's shell history.
    """
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        **fields,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
