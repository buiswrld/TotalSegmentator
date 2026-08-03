"""
Stage 4 of the QC pipeline: turn one split's combined_metrics.csv (stage 2 output) into
a clean, ready-to-train dataset - joining the organ-relative z-score features (stage 3's
reference table), dropping empty-prediction rows, and assigning the accept/reject
training label. Run this once per split (e.g. once for val's combined_metrics.csv, once
for test's) to produce val.csv / test.csv independently.

This used to be inlined at the top of train_classifier.py's main() - pulling it into its
own stage means the exact dataset a model trains/tests on can be inspected, re-generated,
or handed to someone else without re-running inference or scoring.

Writes --output-csv: ID_COLS (Subject ID, Organ/Structure Name, Original Image Axis
Orientation Codes, Predicted Mask Voxel Count) + Intersection over Union (IoU, kept for
reporting/inspection only - never selected as a model feature) + every raw
ground-truth-free feature + every organ-relative z-score feature (feature name + "_z"
suffix) + the Accept/Reject Training Label (label) column. See
totalsegmentator/qc_columns.py for the glossary.

Run:  python experiments/pipeline/curate_dataset.py --metrics-csv metrics/val_combined_metrics.csv --reference-csv reference/reference_stats.csv --output-csv datasets/val.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.build_reference_table import load_reference, add_relative_features
from experiments.pipeline.common import write_manifest
from totalsegmentator.qc_columns import (
    COL_SUBJECT, COL_ORGAN, COL_ORIG_AXCODES, COL_PREDICTED_VOXEL_COUNT,
    COL_DICE, COL_IOU, COL_TRUE_POSITIVES, COL_FALSE_POSITIVES, COL_FALSE_NEGATIVES,
    COL_GROUND_TRUTH_VOXEL_COUNT, COL_MATCH_STATUS, COL_ACCEPT_LABEL, COL_TRAINING_LABEL,
    COL_IS_EMPTY,
)

# Kept in the curated output for reporting/inspection (e.g. oof_predictions.csv /
# test_results.csv show predicted probability next to actual IoU) even though it's a
# LEAK_COLS entry and therefore never selected as a model feature - see curate()'s
# raw_cols/rel_cols construction below, which still excludes it correctly.
DIAGNOSTIC_COLS = [COL_IOU]

# Anything computed from ground truth. Using these as features would leak the label.
LEAK_COLS = [COL_DICE, COL_IOU, COL_TRUE_POSITIVES, COL_FALSE_POSITIVES, COL_FALSE_NEGATIVES,
             COL_GROUND_TRUTH_VOXEL_COUNT, COL_MATCH_STATUS, COL_ACCEPT_LABEL]
ID_COLS = [COL_SUBJECT, COL_ORGAN, COL_ORIG_AXCODES, COL_PREDICTED_VOXEL_COUNT]


def curate(metrics_csv, reference_csv, iou_accept, drop_empty):
    """(df, raw_cols, rel_cols) - the curated dataframe and its feature-column lists."""
    df = pd.read_csv(metrics_csv)
    df[COL_TRAINING_LABEL] = (df[COL_IOU] >= iou_accept).astype(int)

    n0 = len(df)
    if drop_empty and COL_IS_EMPTY in df:
        df = df[df[COL_IS_EMPTY] == 0].copy()
        print(f"dropped {n0 - len(df)} empty-prediction rows "
              f"(no mask to judge; a rule handles these, not the model)")

    raw_cols = [c for c in df.columns if c not in LEAK_COLS + ID_COLS + [COL_TRAINING_LABEL]]

    rel_cols = []
    if reference_csv and os.path.exists(reference_csv):
        ref = load_reference(reference_csv)
        df = add_relative_features(df, ref)
        rel_cols = [c for c in df.columns if c.endswith("_z")]
        covered = df[COL_ORGAN].isin(ref.keys()).mean()
        print(f"joined reference table: {len(rel_cols)} organ-relative features, "
              f"{len(ref)} organs in table, {covered*100:.1f}% of rows covered")
    elif reference_csv:
        print(f"NOTE: {reference_csv} not found - skipping organ-relative features")

    assert not set(raw_cols + rel_cols) & set(LEAK_COLS), "ground-truth column leaked"
    return df, raw_cols, rel_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-csv", required=True, help="combined_metrics.csv from stage 2, for one split.")
    parser.add_argument("--reference-csv", default=None,
                        help="reference_stats.csv from stage 3. Omit to skip organ-relative z-score features.")
    parser.add_argument("--output-csv", required=True, help="Where to write the curated dataset.")
    parser.add_argument("--iou-accept", type=float, default=0.90, help="IoU threshold for the accept/reject label.")
    parser.add_argument("--keep-empty", action="store_true",
                        help="Keep empty-prediction rows instead of dropping them (default: drop).")
    args = parser.parse_args()

    df, raw_cols, rel_cols = curate(args.metrics_csv, args.reference_csv, args.iou_accept, not args.keep_empty)

    cols = ID_COLS + DIAGNOSTIC_COLS + raw_cols + rel_cols + [COL_TRAINING_LABEL]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    df[cols].to_csv(args.output_csv, index=False)

    write_manifest(
        os.path.splitext(args.output_csv)[0] + "_manifest.json",
        stage="curate_dataset", metrics_csv=args.metrics_csv, reference_csv=args.reference_csv,
        iou_accept=args.iou_accept, drop_empty=not args.keep_empty,
        n_rows=len(df), n_raw_features=len(raw_cols), n_relative_features=len(rel_cols),
        raw_feature_columns=raw_cols, relative_feature_columns=rel_cols,
    )
    print(f"\ncurated dataset ({len(df)} rows, {len(cols)} columns) -> {args.output_csv}")


if __name__ == "__main__":
    main()
