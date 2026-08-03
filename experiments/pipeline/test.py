"""
Stage 6 of the QC pipeline: score persisted model(s) from stage 5 against a real held-out
curated test set (stage 4 output, built from the dataset's own `test` split - never
touched by stage 3's reference table or stage 5's training/CV).

This is the stage that didn't exist before this pipeline was split up: previously
train_classifier.py only ever reported cross-validated numbers, and calibrate.py only
ever did nested CV - neither ever fit-saved-reloaded-and-scored a model against data it
was never trained or cross-validated on.

For every model in --models-dir (or just --model-name, if given): loads the persisted
sklearn Pipeline + its recorded feature-column list from training_manifest.json, scores
it against --test-csv, and reports ROC-AUC/PR-AUC plus the same calibration metrics as
the old calibrate.py (ECE/MCE/Brier decomposition) - ranking quality AND whether the
model's scores are usable as real probabilities.

Writes:
  <output-dir>/test_results.csv   one row per (subject, organ): true label + one
                                   predicted-probability column per model
  <output-dir>/test_summary.csv   one row per model: ROC-AUC, PR-AUC(accept/reject),
                                   ECE, MCE, Brier score + its reliability/resolution/
                                   uncertainty decomposition

Run:  python experiments/pipeline/test.py --test-csv datasets/test.csv --models-dir models --output-dir results/test
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.common import ece_mce, brier_decomposition, write_manifest
from totalsegmentator.qc_columns import COL_SUBJECT, COL_ORGAN, COL_IOU, COL_TRAINING_LABEL


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-csv", required=True, help="Curated held-out test set from stage 4.")
    parser.add_argument("--models-dir", required=True, help="Directory of .joblib models + training_manifest.json from stage 5.")
    parser.add_argument("--model-name", default=None,
                        help="Score only this one model (its display name, e.g. 'randomforest [raw+rel]'). Default: every persisted model.")
    parser.add_argument("--output-dir", required=True, help="Where test_results.csv/test_summary.csv go.")
    args = parser.parse_args()

    manifest_path = os.path.join(args.models_dir, "training_manifest.json")
    with open(manifest_path) as fh:
        training_manifest = json.load(fh)
    models = training_manifest["models"]
    if args.model_name:
        if args.model_name not in models:
            raise SystemExit(f"'{args.model_name}' not found in {manifest_path}. "
                             f"Available: {', '.join(models)}")
        models = {args.model_name: models[args.model_name]}

    df = pd.read_csv(args.test_csv)
    y = df[COL_TRAINING_LABEL].to_numpy()
    print(f"{len(df)} masks | {df[COL_SUBJECT].nunique()} subjects | "
          f"{y.mean()*100:.1f}% accepted (positive class) | {len(models)} model(s) to test\n")

    results_df = df[[COL_SUBJECT, COL_ORGAN, COL_IOU, COL_TRAINING_LABEL]].copy()
    summary_rows = []
    for name, info in models.items():
        pipe = joblib.load(info["path"])
        cols = info["feature_columns"] + [COL_ORGAN]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  {name}: SKIPPED - test set is missing feature columns {missing}")
            continue
        p = pipe.predict_proba(df[cols])[:, 1]
        results_df[name.replace(" ", "_").replace("[", "").replace("]", "")] = p

        auc = roc_auc_score(y, p)
        pr_auc_accept = average_precision_score(y, p)
        pr_auc_reject = average_precision_score(1 - y, 1 - p)
        ece, mce, _ = ece_mce(y, p)
        bd = brier_decomposition(y, p)
        summary_rows.append({"model": name, "n_test_rows": len(df), "roc_auc": auc,
                             "pr_auc_accept": pr_auc_accept, "pr_auc_reject": pr_auc_reject,
                             "ece": ece, "mce": mce, **bd})
        print(f"{name:32s} ROC-AUC {auc:.3f}  PR-AUC(acc) {pr_auc_accept:.3f}  "
              f"PR-AUC(rej) {pr_auc_reject:.3f}  ECE {ece:.3f}  MCE {mce:.3f}  Brier {bd['brier']:.3f}")

    if not summary_rows:
        raise SystemExit("No models could be scored - check --test-csv has the expected feature columns.")

    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(args.output_dir, "test_results.csv"), index=False)
    summary = pd.DataFrame(summary_rows).sort_values("pr_auc_reject", ascending=False)
    summary.to_csv(os.path.join(args.output_dir, "test_summary.csv"), index=False)

    write_manifest(
        os.path.join(args.output_dir, "test_manifest.json"),
        stage="test", test_csv=args.test_csv, models_dir=args.models_dir,
        model_name=args.model_name, n_test_rows=len(df), n_models_scored=len(summary_rows),
        train_csv=training_manifest.get("train_csv"),
    )
    print(f"\ntest_results.csv + test_summary.csv -> {args.output_dir}")
    print(f"best on held-out test (by PR-AUC reject): {summary.iloc[0]['model']}")


if __name__ == "__main__":
    main()
