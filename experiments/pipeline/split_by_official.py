"""
Regroup already-computed stage-2 metrics by TotalSegmentator's own official
train/val/test split, instead of this pipeline's pooled 80:20 partition.

Why this exists: `experiments/eval_runs/{mri,ct}_full_remote/metrics/` holds
combined_metrics.csv split by our pooled "classifier_train"/"classifier_test" roles
(subjects pooled and shuffled independent of meta.csv's split column - see
common.py::get_subjects, split="all"). This script does NOT re-run inference or
re-score anything; it just concatenates those pieces back together and re-splits the
rows by each subject's real meta.csv split label, producing input files
curate_dataset.py (stage 4) can consume unmodified.

CT's official split has three parts (train/val/test); --ct-val-into treats val as part
of train, test, or drops it (default: test, matching the GRAM paper's framing where CT
val is the primary held-out set). MR only has train/test.

Prints a coverage report (found vs. expected per split) rather than failing silently -
some subjects' predictions never made it into a checkpointed combined_metrics.csv
before the source GPU box was decommissioned, so 100% coverage is not expected.

Run:
  python experiments/pipeline/split_by_official.py \
    --dataset-dir /Volumes/Datasets/Totalsegmentator_dataset_v201 --modality ct \
    --metrics-csv experiments/eval_runs/ct_full_remote/metrics/classifier_train/combined_metrics.csv \
    --metrics-csv experiments/eval_runs/ct_full_remote/metrics/classifier_test/combined_metrics.csv \
    --output-dir experiments/eval_runs/ct_official_split/metrics
"""

import argparse
import csv
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.common import find_columns, write_manifest
from totalsegmentator.qc_columns import COL_SUBJECT


def load_official_splits(dataset_dir):
    """{subject_id: split_label} straight from meta.csv - the real, authoritative
    train/val/test assignment, not this pipeline's pooled scheme."""
    meta = os.path.join(dataset_dir, "meta.csv")
    with open(meta, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096); fh.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(fh, delimiter=delim)
        id_col, split_col = find_columns(reader.fieldnames or [])
        if not (id_col and split_col):
            raise SystemExit(f"Could not find id/split columns in {meta} (got {reader.fieldnames})")
        return {row[id_col].strip(): row[split_col].strip().lower() for row in reader}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, help="Dataset root containing meta.csv.")
    p.add_argument("--metrics-csv", action="append", required=True,
                    help="A stage-2 combined_metrics.csv to include; repeatable (e.g. both the old "
                         "classifier_train and classifier_test pieces).")
    p.add_argument("--output-dir", required=True,
                    help="Where official_train_combined_metrics.csv / official_test_combined_metrics.csv go.")
    p.add_argument("--ct-val-into", choices=["test", "train", "drop"], default="test",
                    help="Only relevant if meta.csv has a 'val' split (CT): fold it into official-test "
                         "(default, matches the GRAM paper's framing), official-train, or drop it.")
    args = p.parse_args()

    split_of = load_official_splits(args.dataset_dir)
    expected = Counter(split_of.values())

    df = pd.concat([pd.read_csv(f) for f in args.metrics_csv], ignore_index=True)
    n_before = len(df)
    dupe_subjects = df[df.duplicated(subset=[COL_SUBJECT, df.columns[1]], keep=False)][COL_SUBJECT].unique()
    if len(dupe_subjects):
        raise SystemExit(f"Unexpected duplicate (subject, organ) rows across --metrics-csv inputs "
                          f"for {len(dupe_subjects)} subject(s), e.g. {list(dupe_subjects[:5])} - "
                          f"were the same pooled piece passed twice?")

    def official_role(subject):
        s = split_of.get(subject)
        if s == "val":
            return {"test": "test", "train": "train", "drop": None}[args.ct_val_into]
        return s  # "train" / "test" / None if the subject isn't in meta.csv at all

    df["_official_role"] = df[COL_SUBJECT].map(official_role)
    unmatched = sorted(df.loc[df["_official_role"].isna(), COL_SUBJECT].unique())
    if unmatched:
        print(f"  warning: {len(unmatched)} subject(s) in --metrics-csv not found in meta.csv "
              f"(or dropped by --ct-val-into=drop), excluded: {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'role':<12} {'found':>8} {'expected':>10}   coverage")
    for role in ["train", "test"]:
        role_df = df[df["_official_role"] == role].drop(columns=["_official_role"])
        n_subjects = role_df[COL_SUBJECT].nunique()
        n_expected = expected.get(role, 0) + (expected.get("val", 0) if role == args.ct_val_into else 0)
        out_path = os.path.join(args.output_dir, f"official_{role}_combined_metrics.csv")
        role_df.to_csv(out_path, index=False)
        print(f"{role:<12} {n_subjects:>8} {n_expected:>10}   {n_subjects/n_expected:.1%}  -> {out_path}")

    write_manifest(
        os.path.join(args.output_dir, "split_by_official_manifest.json"),
        dataset_dir=args.dataset_dir, metrics_csv=args.metrics_csv, ct_val_into=args.ct_val_into,
        rows_in=n_before, expected_by_split=dict(expected), unmatched_subjects=unmatched,
    )


if __name__ == "__main__":
    main()
