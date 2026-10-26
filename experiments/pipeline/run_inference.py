"""
Stage 1 of the QC pipeline: run TotalSegmentator over a dataset split, producing
predictions on disk. Ground-truth-free, model-only - this stage never looks at
segmentations/.

Idempotent/resumable: a subject already having prediction files in --predictions-dir is
skipped, so re-running after an interrupted run only does the missing subjects. Pass
--force to wipe and redo a subject's predictions.

--num-shards/--shard-index split a split's subject list across N parallel processes
(e.g. one per GPU) via subjects[shard_index::num_shards] - disjoint by construction, so
any number of shards can safely write into the same --predictions-dir concurrently.
Default is a single unsharded run (num_shards=1, shard_index=0), unchanged from before.

Writes:
  <predictions-dir>/<subject>/*.nii.gz   one folder per subject (TotalSegmentator's own
                                          per-structure or --ml multilabel output)
  <predictions-dir>/manifest.json        which subjects/modality/task/args produced this
                                          shard's run (one per shard - see --manifest-name)

Run:  python experiments/pipeline/run_inference.py --dataset-dir <dir> --predictions-dir <dir> --modality mr --split test
Sharded (GPU 3 of 8):
      python experiments/pipeline/run_inference.py --dataset-dir <dir> --predictions-dir <dir> \
        --modality ct --split train --device cuda:3 --num-shards 8 --shard-index 3
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
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split this split's subjects across N parallel processes (e.g. one per GPU).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard this process handles, 0..num-shards-1.")
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}) but got {args.shard_index}")

    image_name, task = resolve_modality(args)
    subjects = get_subjects(args.dataset_dir, args.split, args.limit, args.offset)
    if args.num_shards > 1:
        subjects = subjects[args.shard_index::args.num_shards]
    shard_tag = f" | shard {args.shard_index}/{args.num_shards}" if args.num_shards > 1 else ""
    print(f"Running inference for {len(subjects)} subject(s) | modality={args.modality} "
          f"task={task} image_name={image_name}{shard_tag}\n")

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

    manifest_name = f"manifest_shard{args.shard_index}.json" if args.num_shards > 1 else "manifest.json"
    write_manifest(
        os.path.join(args.predictions_dir, manifest_name),
        stage="run_inference", dataset_dir=args.dataset_dir, modality=args.modality,
        image_name=image_name, task=task, split=args.split, limit=args.limit,
        device=args.device, extra_args=args.extra_args,
        num_shards=args.num_shards, shard_index=args.shard_index,
        n_subjects=len(subjects), n_completed=len(completed), n_skipped=len(skipped),
        n_failed=len(failed), failed_subjects=failed,
    )
    print(f"\n{len(completed)} run, {len(skipped)} already present, {len(failed)} failed/missing")
    print(f"Predictions written to {args.predictions_dir}")


if __name__ == "__main__":
    main()
