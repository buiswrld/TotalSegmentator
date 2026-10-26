"""
Stage 5 of the QC pipeline: train the mask accept/reject classifier on a curated
dataset (stage 4 output).

Two things happen:

  1. Cross-validated model search - compares feature sets and model architectures under
     subject-grouped CV, same three questions as before:
       - Does anything beat a trivial baseline?
       - Does the model learn from mask GEOMETRY, or is it just memorising which organ
         is easy? (the organ-rate baseline is the control for this)
       - Do organ-relative features let the signal transfer to structures the model has
         never seen? (the leave-one-organ-out experiment)
  2. NEW: every (model family x feature set) pipeline is then refit on the FULL curated
     training set and persisted to --models-dir as a .joblib file, individually - so
     stage 6 (test.py) can load any one of them and score it against a real held-out test
     set, rather than only ever reporting cross-validated numbers.

Writes:
  <output-dir>/oof_predictions.csv     out-of-fold predictions from the CV comparison
  <output-dir>/loo_by_organ.csv        leave-one-organ-out results (if --run-loo)
  <output-dir>/cv_summary.csv          ROC-AUC/PR-AUC per model, from the CV comparison
  <models-dir>/<model_name>.joblib     one persisted sklearn Pipeline per model+feature-set
  <models-dir>/training_manifest.json  which feature columns/args/CV scores produced each model

Run:  python experiments/pipeline/train.py --train-csv datasets/val.csv --output-dir results/train --models-dir models
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.common import write_manifest
from totalsegmentator.qc_columns import COL_SUBJECT, COL_ORGAN, COL_IOU, COL_TRAINING_LABEL

# Raw features that already mean the same thing for every structure, so they need no
# per-organ reference - largest_component_fraction in particular is the single
# strongest feature and is organ-independent by construction.
ORGAN_FREE_RAW_HINTS = ("Number of Connected Components", "Largest Connected Component Fraction",
                        "Touches Image Boundary", "Boundary Voxel Fraction")


def safe_name(name):
    return name.replace(" ", "_").replace("[", "").replace("]", "").replace("+", "-")


def resolve_feature_columns(train_csv):
    """Recover the raw/relative feature-column split written by curate_dataset.py's
    manifest, falling back to a naming-convention heuristic (any "*_z" column is
    organ-relative) if the manifest isn't next to --train-csv.
    """
    manifest_path = os.path.splitext(train_csv)[0] + "_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            m = json.load(fh)
        return m["raw_feature_columns"], m["relative_feature_columns"]
    df_cols = pd.read_csv(train_csv, nrows=0).columns.tolist()
    id_and_label = {COL_SUBJECT, COL_ORGAN, COL_TRAINING_LABEL}
    rel_cols = [c for c in df_cols if c.endswith("_z")]
    raw_cols = [c for c in df_cols if c not in id_and_label and c not in rel_cols
                and not c.startswith("Original Image Axis") and not c.startswith("Predicted Mask Voxel")]
    print(f"NOTE: {manifest_path} not found - recovered {len(raw_cols)} raw / {len(rel_cols)} "
          f"relative feature columns by naming convention instead.")
    return raw_cols, rel_cols


def make_feature_sets(raw_cols, rel_cols):
    """{name: (numeric_columns, use_organ_onehot)}"""
    sets = {"raw": (raw_cols, False), "raw+organ": (raw_cols, True)}
    if rel_cols:
        organ_free = [c for c in raw_cols if c.startswith(ORGAN_FREE_RAW_HINTS)]
        sets["rel"] = (rel_cols + organ_free, False)
        sets["raw+rel"] = (raw_cols + rel_cols, False)
    return sets


def build_models(feature_sets, seed):
    """[(display_name, pipeline), ...] across every model family x feature set."""
    def pre(cols, use_organ, scaled):
        num = Pipeline([("imp", SimpleImputer(strategy="median"))] +
                       ([("sc", StandardScaler())] if scaled else []))
        parts = [("num", num, cols)]
        if use_organ:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [COL_ORGAN]))
        return ColumnTransformer(parts)

    families = [
        ("logreg", lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), True),
        ("tree-depth3", lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed), False),
        ("randomforest", lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                        class_weight="balanced_subsample",
                                                        random_state=seed, n_jobs=-1), False),
        ("histgradboost", lambda: HistGradientBoostingClassifier(max_iter=300, random_state=seed), False),
    ]
    out = []
    for set_name, (cols, use_organ) in feature_sets.items():
        for fam, make_clf, scaled in families:
            out.append((f"{fam} [{set_name}]", cols,
                        Pipeline([("pre", pre(cols, use_organ, scaled)), ("clf", make_clf())])))
    return out


def organ_baseline(train_df, test_df, global_rate):
    """Predict each organ's accept rate, learned from training folds only. Uses NO mask
    geometry - a model that cannot beat this has learned nothing except which organs
    are easy, which is a lookup table rather than a quality check."""
    rates = train_df.groupby(COL_ORGAN)[COL_TRAINING_LABEL].mean()
    return test_df[COL_ORGAN].map(rates).fillna(global_rate).values


def evaluate(df, models, n_splits, output_dir):
    y = df[COL_TRAINING_LABEL].values
    groups = df[COL_SUBJECT].values
    print(f"\n{len(df)} masks | {df[COL_SUBJECT].nunique()} subjects | "
          f"{y.mean()*100:.1f}% accepted (positive class)")

    cv = GroupKFold(n_splits=min(n_splits, df[COL_SUBJECT].nunique()))
    scores = {name: {"auc": [], "ap": []} for name, _, _ in models}
    scores["organ-rate baseline"] = {"auc": [], "ap": []}
    scores["always-accept"] = {"auc": [], "ap": []}
    oof = {name: np.zeros(len(df)) for name in scores}

    for tr, te in cv.split(df, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        tr_df, te_df = df.iloc[tr], df.iloc[te]

        p = organ_baseline(tr_df, te_df, y[tr].mean())
        scores["organ-rate baseline"]["auc"].append(roc_auc_score(y[te], p))
        scores["organ-rate baseline"]["ap"].append(average_precision_score(y[te], p))
        oof["organ-rate baseline"][te] = p

        p = np.full(len(te), y[tr].mean())
        scores["always-accept"]["auc"].append(0.5)
        scores["always-accept"]["ap"].append(average_precision_score(y[te], p))
        oof["always-accept"][te] = p

        for name, cols, pipe in models:
            pipe.fit(tr_df[cols + [COL_ORGAN]], y[tr])
            p = pipe.predict_proba(te_df[cols + [COL_ORGAN]])[:, 1]
            scores[name]["auc"].append(roc_auc_score(y[te], p))
            scores[name]["ap"].append(average_precision_score(y[te], p))
            oof[name][te] = p

    print(f"\n{'model':32s} {'ROC-AUC':>16s} {'PR-AUC(acc)':>16s}")
    print("-" * 68)
    ranked = sorted(scores.items(), key=lambda kv: -np.mean(kv[1]["ap"] or [0]))
    summary_rows = []
    for name, s in ranked:
        if not s["ap"]:
            continue
        print(f"{name:32s} {np.mean(s['auc']):>8.3f} +/-{np.std(s['auc']):<6.3f} "
              f"{np.mean(s['ap']):>8.3f} +/-{np.std(s['ap']):<6.3f}")
        summary_rows.append({"model": name, "roc_auc_mean": np.mean(s["auc"]), "roc_auc_std": np.std(s["auc"]),
                             "pr_auc_mean": np.mean(s["ap"]), "pr_auc_std": np.std(s["ap"])})

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "cv_summary.csv"), index=False)

    out = df[[COL_SUBJECT, COL_ORGAN, COL_IOU, COL_TRAINING_LABEL]].copy()
    for name in oof:
        out[safe_name(name)] = oof[name]
    out.to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)
    print(f"\ncv_summary.csv + oof_predictions.csv -> {output_dir}")
    return ranked[0][0]


def leave_one_organ_out(df, feature_sets, min_n, min_minority, seed, output_dir):
    """Train on every organ EXCEPT one, test on the held-out organ - tests whether the
    quality signal is structural or memorised (organ-relative z-scores should transfer
    to an unseen organ; raw features cannot)."""
    usable = []
    for o, g in df.groupby(COL_ORGAN):
        n_pos, n_neg = int(g[COL_TRAINING_LABEL].sum()), int((1 - g[COL_TRAINING_LABEL]).sum())
        if len(g) >= min_n and min(n_pos, n_neg) >= min_minority:
            usable.append((o, g))
    print(f"\n=== leave-one-organ-out ({len(usable)} organs, >= {min_n} rows each) ===")

    sets = {k: v for k, v in feature_sets.items() if not v[1]}  # no organ one-hot in LOO
    results = {k: [] for k in sets}
    per_organ = {}
    for organ, test_df in usable:
        train_df = df[df[COL_ORGAN] != organ]
        row = {}
        for set_name, (cols, _) in sets.items():
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight="balanced_subsample", random_state=seed, n_jobs=-1))])
            pipe.fit(train_df[cols], train_df[COL_TRAINING_LABEL])
            p = pipe.predict_proba(test_df[cols])[:, 1]
            auc = roc_auc_score(test_df[COL_TRAINING_LABEL], p)
            results[set_name].append(auc)
            row[set_name] = auc
        per_organ[organ] = (len(test_df), test_df[COL_TRAINING_LABEL].mean(), row)

    print(f"{'MEAN':30s} " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in results.items() if v))
    out = pd.DataFrame([{COL_ORGAN: o, "n": per_organ[o][0], "accept_rate": per_organ[o][1], **per_organ[o][2]}
                        for o in per_organ])
    out.to_csv(os.path.join(output_dir, "loo_by_organ.csv"), index=False)
    print(f"loo_by_organ.csv -> {output_dir}")


def persist_models(df, models, models_dir, cv_scores_by_name, args):
    """Refit every model+feature-set pipeline on the FULL curated training set and save
    it individually, so stage 6 can load and test any one of them (or all of them, for
    a combined comparison) against a real held-out test set.
    """
    os.makedirs(models_dir, exist_ok=True)
    y = df[COL_TRAINING_LABEL].values
    persisted = {}
    for name, cols, pipe in models:
        pipe.fit(df[cols + [COL_ORGAN]], y)
        model_path = os.path.join(models_dir, safe_name(name) + ".joblib")
        joblib.dump(pipe, model_path)
        persisted[name] = {"path": model_path, "feature_columns": cols, "uses_organ_onehot": True}
        print(f"  saved {name:32s} -> {model_path}")

    write_manifest(
        os.path.join(models_dir, "training_manifest.json"),
        stage="train", train_csv=args.train_csv, n_splits=args.n_splits, seed=args.seed,
        run_loo=not args.skip_loo, n_rows=len(df), n_subjects=int(df[COL_SUBJECT].nunique()),
        models=persisted, cv_scores=cv_scores_by_name,
    )
    print(f"\ntraining_manifest.json -> {models_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", required=True, help="Curated dataset from stage 4 (curate_dataset.py).")
    parser.add_argument("--output-dir", required=True, help="Where cv_summary.csv/oof_predictions.csv/loo_by_organ.csv go.")
    parser.add_argument("--models-dir", required=True, help="Where persisted .joblib models + training_manifest.json go.")
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits for the CV comparison.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-loo", action="store_true", help="Skip the leave-one-organ-out experiment.")
    parser.add_argument("--loo-min-n", type=int, default=25, help="Rows needed for an organ to enter LOO.")
    parser.add_argument("--loo-min-minority", type=int, default=5, help="Minimum rarer-class count for a usable LOO AUC.")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    raw_cols, rel_cols = resolve_feature_columns(args.train_csv)
    feature_sets = make_feature_sets(raw_cols, rel_cols)
    print("feature sets: " + ", ".join(f"{k} ({len(v[0])}{'+organ' if v[1] else ''})" for k, v in feature_sets.items()))

    models = build_models(feature_sets, args.seed)
    best_name = evaluate(df, models, args.n_splits, args.output_dir)
    print(f"\nbest model (CV): {best_name}")

    if not args.skip_loo and rel_cols:
        leave_one_organ_out(df, feature_sets, args.loo_min_n, args.loo_min_minority, args.seed, args.output_dir)

    cv_summary = pd.read_csv(os.path.join(args.output_dir, "cv_summary.csv")).set_index("model")
    cv_scores_by_name = {n: {"roc_auc_mean": float(cv_summary.loc[n, "roc_auc_mean"])}
                         for n, _, _ in models if n in cv_summary.index}
    print(f"\n=== persisting models (refit on full {len(df)}-row training set) ===")
    persist_models(df, models, args.models_dir, cv_scores_by_name, args)


if __name__ == "__main__":
    main()
