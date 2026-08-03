"""
Convenience orchestrator for the 6-stage QC pipeline. Computes the standard sub-paths
under --run-dir and invokes each stage script in order via subprocess:

  run-dir/
    predictions/<split>/<subject>/*.nii.gz     stage 1, once per split that needs it
    metrics/<split>/combined_metrics.csv, ...  stage 2, once per split that needs it
    reference/reference_stats.csv              stage 3 (built from --reference-split)
    datasets/classifier_train.csv              stage 4, curated from --train-split
    datasets/classifier_test.csv               stage 4, curated from --test-split
    models/<name>.joblib, training_manifest.json  stage 5
    results/train/  results/test/              stage 5 / stage 6 reports

Every stage remains fully independently runnable with its own explicit paths (see each
script's own --help) - this wrapper is only for running the whole thing at once with one
consistent directory layout. Stages are skipped if their output already exists, unless
--force is passed (mirrors stage 1's own per-subject prediction caching, applied
pipeline-wide).

--reference-split (default: train) builds the organ-relative reference table.
--train-split (default: val) is what the classifier trains/cross-validates on.
--test-split (default: test) is held out end-to-end for stage 6's final evaluation.
This matches the CT dataset's own train/val/test split; for a dataset without a val
split (e.g. the MR benchmark dataset, which only has train/test), pass
--train-split test --skip-stage 6 (there is nothing left to hold out), or curate your
own val subset first.

Run:  python experiments/pipeline/run_pipeline.py --run-dir runs/ct_v1 --dataset-dir <dir> --modality ct
Run a subset:  python experiments/pipeline/run_pipeline.py --run-dir runs/ct_v1 --dataset-dir <dir> --modality ct --stages 4,5,6
"""

import argparse
import os
import subprocess
import sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script, *cli_args):
    cmd = [sys.executable, os.path.join(PIPELINE_DIR, script)] + [str(a) for a in cli_args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def exists(path):
    return os.path.exists(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Root directory for every stage's output.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--modality", choices=["ct", "mr"], default="ct")
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default=None, help="TotalSegmentator -d flag for stage 1.")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--train-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Applied to every split.")
    parser.add_argument("--reference-limit", type=int, default=200, help="Subjects for stage 3 (default 200).")
    parser.add_argument("--min-voxels", type=int, default=20)
    parser.add_argument("--dice-flag", type=float, default=0.50)
    parser.add_argument("--iou-accept", type=float, default=0.90)
    parser.add_argument("--reference-workers", type=int, default=10)
    parser.add_argument("--expected-dice-json", default=None,
                        help="e.g. resources/expected_dice_mr.json - only meaningful for --modality mr.")
    parser.add_argument("--n-splits", type=int, default=5, help="stage 5 CV folds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--stages", default="1,2,3,4,5,6",
                        help="Comma-separated stage numbers to run (default: all six).")
    parser.add_argument("--force", action="store_true", help="Re-run stages even if their output already exists.")
    args = parser.parse_args()
    stages = {int(s) for s in args.stages.split(",")}

    predictions_dir = lambda split: os.path.join(args.run_dir, "predictions", split)
    metrics_dir = lambda split: os.path.join(args.run_dir, "metrics", split)
    reference_csv = os.path.join(args.run_dir, "reference", "reference_stats.csv")
    dataset_csv = lambda role: os.path.join(args.run_dir, "datasets", f"classifier_{role}.csv")
    models_dir = os.path.join(args.run_dir, "models")
    results_dir = lambda role: os.path.join(args.run_dir, "results", role)

    def modality_args():
        a = ["--modality", args.modality]
        if args.image_name: a += ["--image-name", args.image_name]
        if args.task: a += ["--task", args.task]
        return a

    # ---- stage 1 + 2, once per split that feeds the classifier (train-split, test-split) ----
    for role, split in (("train", args.train_split), ("test", args.test_split)):
        combined_csv = os.path.join(metrics_dir(split), "combined_metrics.csv")

        if 1 in stages:
            if args.force or not exists(predictions_dir(split)):
                cli = ["--dataset-dir", args.dataset_dir, "--predictions-dir", predictions_dir(split),
                      "--split", split] + modality_args()
                if args.limit: cli += ["--limit", args.limit]
                if args.device: cli += ["--device", args.device]
                if args.force: cli += ["--force"]
                run("run_inference.py", *cli)
            else:
                print(f"\n[skip] stage 1 ({split}) - {predictions_dir(split)} already exists")

        if 2 in stages:
            if args.force or not exists(combined_csv):
                cli = ["--dataset-dir", args.dataset_dir, "--predictions-dir", predictions_dir(split),
                      "--split", split, "--output-csv", combined_csv,
                      "--min-voxels", args.min_voxels, "--dice-flag", args.dice_flag,
                      "--iou-accept", args.iou_accept] + modality_args()
                if args.limit: cli += ["--limit", args.limit]
                if args.expected_dice_json: cli += ["--expected-dice-json", args.expected_dice_json]
                run("compute_metrics.py", *cli)
            else:
                print(f"\n[skip] stage 2 ({split}) - {combined_csv} already exists")

    # ---- stage 3: reference table, from --reference-split only ----
    if 3 in stages:
        if args.force or not exists(reference_csv):
            cli = ["--dataset-dir", args.dataset_dir, "--split", args.reference_split,
                  "--output-csv", reference_csv, "--workers", args.reference_workers,
                  "--min-voxels", args.min_voxels, "--limit", args.reference_limit] + modality_args()
            run("build_reference_table.py", *cli)
        else:
            print(f"\n[skip] stage 3 - {reference_csv} already exists")

    # ---- stage 4: curate train-split and test-split into classifier-ready CSVs ----
    if 4 in stages:
        for role, split in (("train", args.train_split), ("test", args.test_split)):
            out_csv = dataset_csv(role)
            if args.force or not exists(out_csv):
                combined_csv = os.path.join(metrics_dir(split), "combined_metrics.csv")
                run("curate_dataset.py", "--metrics-csv", combined_csv, "--reference-csv", reference_csv,
                   "--output-csv", out_csv, "--iou-accept", args.iou_accept)
            else:
                print(f"\n[skip] stage 4 ({role}) - {out_csv} already exists")

    # ---- stage 5: train + persist models ----
    if 5 in stages:
        if args.force or not exists(os.path.join(models_dir, "training_manifest.json")):
            cli = ["--train-csv", dataset_csv("train"), "--output-dir", results_dir("train"),
                  "--models-dir", models_dir, "--n-splits", args.n_splits, "--seed", args.seed]
            if args.skip_loo: cli += ["--skip-loo"]
            run("train.py", *cli)
        else:
            print(f"\n[skip] stage 5 - {models_dir}/training_manifest.json already exists")

    # ---- stage 6: score persisted models against the held-out test set ----
    if 6 in stages:
        if args.force or not exists(os.path.join(results_dir("test"), "test_summary.csv")):
            run("test.py", "--test-csv", dataset_csv("test"), "--models-dir", models_dir,
               "--output-dir", results_dir("test"))
        else:
            print(f"\n[skip] stage 6 - {results_dir('test')}/test_summary.csv already exists")

    print(f"\nPipeline complete. Outputs under {args.run_dir}")


if __name__ == "__main__":
    main()
