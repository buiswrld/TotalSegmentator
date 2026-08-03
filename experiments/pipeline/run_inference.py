"""
Stage 1 of the QC pipeline: run TotalSegmentator over a dataset split, producing
predictions on disk. Ground-truth-free, model-only - this stage never looks at
segmentations/.

Idempotent/resumable: a subject already having prediction files in --predictions-dir is
skipped, so re-running after an interrupted run only does the missing subjects. Pass
--force to wipe and redo a subject's predictions.

Writes:
  <predictions-dir>/<subject>/*.nii.gz   one folder per subject (TotalSegmentator's own
                                          per-structure or --ml multilabel output)
  <predictions-dir>/manifest.json        which subjects/modality/task/args produced this run

Run:  python experiments/pipeline/run_inference.py --dataset-dir <dir> --predictions-dir <dir> --modality mr --split test
"""

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.common import (
    add_dataset_args, resolve_modality, get_subjects, ensure_predictions, write_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--predictions-dir", required=True,
                        help="Where to write <subject>/*.nii.gz prediction folders.")
    parser.add_argument("--device", default=None, help="TotalSegmentator -d flag (e.g. cpu, gpu, mps).")
    parser.add_argument("--extra-arg", action="append", default=[], dest="extra_args",
                        help="Extra TotalSegmentator CLI flag, repeatable (e.g. --extra-arg -f --extra-arg --ml).")
    parser.add_argument("--force", action="store_true",
                        help="Delete and regenerate predictions for subjects that already have them.")
    args = parser.parse_args()

    image_name, task = resolve_modality(args)
    subjects = get_subjects(args.dataset_dir, args.split, args.limit)
    print(f"Running inference for {len(subjects)} subject(s) | modality={args.modality} "
          f"task={task} image_name={image_name}\n")

    if args.force:
        for subj in subjects:
            pred_dir = os.path.join(args.predictions_dir, subj)
            if os.path.isdir(pred_dir):
                shutil.rmtree(pred_dir)

    completed, skipped, failed = [], [], []
    for i, subj in enumerate(subjects, 1):
        pred_dir = os.path.join(args.predictions_dir, subj)
        already_had = bool(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
        try:
            result = ensure_predictions(args.dataset_dir, args.predictions_dir, subj,
                                        image_name, task, args.device, args.extra_args)
        except Exception as e:
            print(f"[{i}/{len(subjects)}] {subj}: FAILED ({e})")
            failed.append(subj)
            continue
        if result is None:
            print(f"[{i}/{len(subjects)}] {subj}: missing {image_name}, skipping")
            failed.append(subj)
            continue
        (skipped if already_had else completed).append(subj)
        print(f"[{i}/{len(subjects)}] {subj}: {'already had predictions' if already_had else 'done'}")

    write_manifest(
        os.path.join(args.predictions_dir, "manifest.json"),
        stage="run_inference", dataset_dir=args.dataset_dir, modality=args.modality,
        image_name=image_name, task=task, split=args.split, limit=args.limit,
        device=args.device, extra_args=args.extra_args,
        n_subjects=len(subjects), n_completed=len(completed), n_skipped=len(skipped),
        n_failed=len(failed), failed_subjects=failed,
    )
    print(f"\n{len(completed)} run, {len(skipped)} already present, {len(failed)} failed/missing")
    print(f"Predictions written to {args.predictions_dir}")


if __name__ == "__main__":
    main()
