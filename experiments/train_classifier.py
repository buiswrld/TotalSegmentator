"""
Baseline model search for the mask accept/reject classifier.

Reads combined_metrics.csv (from evaluate_ct.py), optionally joins the per-organ
reference table (from build_reference.py) to add organ-relative z-scores, and compares
feature sets and architectures under subject-grouped cross-validation.

Three questions this answers:

  1. Does anything beat a trivial baseline?
  2. Does the model learn from mask GEOMETRY, or is it just memorising which organ is
     easy? (the organ-rate baseline is the control for this)
  3. Do organ-relative features let the signal transfer to structures the model has
     never seen? (the leave-one-organ-out experiment)

Run:  python train_classifier.py
"""

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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_reference import load_reference, add_relative_features

# ============================ CONFIG ============================
CSV           = r"C:\Users\ansar\Algoverse\report_ct_val\combined_metrics.csv"
REFERENCE_CSV = r"C:\Users\ansar\Algoverse\report_ct\reference_stats.csv"
OUT_DIR       = r"C:\Users\ansar\Algoverse\report_ct_val"

USE_REFERENCE = True   # join reference_stats.csv to add organ-relative z-scores
RUN_LOO       = True   # leave-one-organ-out: train on other organs, test on held-out one
IOU_ACCEPT    = 0.90
N_SPLITS      = 5
DROP_EMPTY    = True
LOO_MIN_N        = 25   # rows needed for the held-out organ
LOO_MIN_MINORITY = 5    # need at least this many of the rarer class for a usable AUC
SEED          = 0
# ================================================================

# Anything computed from ground truth. Using these as features would leak the label.
LEAK_COLS = ["dice", "iou", "tp", "fp", "fn", "gt_vox", "status", "accept"]
ID_COLS   = ["subject", "organ", "orig_axcodes", "pred_vox"]

# Raw features that already mean the same thing for every structure, so they need no
# per-organ reference. These stay in the organ-relative feature set alongside the
# z-scores - largest_component_fraction in particular is the single strongest feature
# and is organ-independent by construction.
ORGAN_FREE_RAW = ["num_components", "largest_component_fraction",
                  "touches_boundary", "boundary_fraction"]


def load(csv_path, iou_accept, drop_empty):
    df = pd.read_csv(csv_path)
    df["label"] = (df["iou"] >= iou_accept).astype(int)

    n0 = len(df)
    if drop_empty and "is_empty" in df:
        df = df[df["is_empty"] == 0].copy()
        print(f"dropped {n0 - len(df)} empty-prediction rows "
              f"(no mask to judge; a rule handles these, not the model)")

    raw_cols = [c for c in df.columns if c not in LEAK_COLS + ID_COLS + ["label"]]

    rel_cols = []
    if USE_REFERENCE:
        if os.path.exists(REFERENCE_CSV):
            ref = load_reference(REFERENCE_CSV)
            df = add_relative_features(df, ref)
            rel_cols = [c for c in df.columns if c.endswith("_z")]
            covered = df["organ"].isin(ref.keys()).mean()
            print(f"joined reference table: {len(rel_cols)} organ-relative features, "
                  f"{len(ref)} organs in table, {covered*100:.1f}% of rows covered")
        else:
            print(f"NOTE: {REFERENCE_CSV} not found - skipping organ-relative features")

    assert not set(raw_cols + rel_cols) & set(LEAK_COLS), "ground-truth column leaked"
    return df, raw_cols, rel_cols


def make_feature_sets(raw_cols, rel_cols):
    """{name: (numeric_columns, use_organ_onehot)}"""
    sets = {"raw": (raw_cols, False),
            "raw+organ": (raw_cols, True)}
    if rel_cols:
        organ_free = [c for c in ORGAN_FREE_RAW if c in raw_cols]
        # 'rel' is organ-agnostic BY CONSTRUCTION: every column means the same thing
        # regardless of structure, which is what makes leave-one-organ-out possible.
        sets["rel"] = (rel_cols + organ_free, False)
        sets["raw+rel"] = (raw_cols + rel_cols, False)
    return sets


def build_models(feature_sets):
    """[(display_name, pipeline), ...] across every model family x feature set."""
    def pre(cols, use_organ, scaled):
        num = Pipeline([("imp", SimpleImputer(strategy="median"))] +
                       ([("sc", StandardScaler())] if scaled else []))
        parts = [("num", num, cols)]
        if use_organ:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False), ["organ"]))
        return ColumnTransformer(parts)

    families = [
        ("logreg",       lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), True),
        ("tree-depth3",  lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                                        random_state=SEED), False),
        ("randomforest", lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                        class_weight="balanced_subsample",
                                                        random_state=SEED, n_jobs=-1), False),
        ("histgradboost", lambda: HistGradientBoostingClassifier(max_iter=300,
                                                                 random_state=SEED), False),
    ]
    out = []
    for set_name, (cols, use_organ) in feature_sets.items():
        for fam, make_clf, scaled in families:
            out.append((f"{fam} [{set_name}]",
                        Pipeline([("pre", pre(cols, use_organ, scaled)),
                                  ("clf", make_clf())])))
    return out


def organ_baseline(train_df, test_df, global_rate):
    """Predict each organ's accept rate, learned from training folds only.

    Uses NO mask geometry. A model that cannot beat this has learned nothing except
    which organs are easy, which is a lookup table rather than a quality check.
    """
    rates = train_df.groupby("organ")["label"].mean()
    return test_df["organ"].map(rates).fillna(global_rate).values


def evaluate(df, feature_sets):
    all_cols = sorted({c for cols, _ in feature_sets.values() for c in cols})
    X = df[all_cols + ["organ"]]
    y = df["label"].values
    groups = df["subject"].values
    print(f"\n{len(df)} masks | {df['subject'].nunique()} subjects | "
          f"{y.mean()*100:.1f}% accepted (positive class)")

    cv = GroupKFold(n_splits=min(N_SPLITS, df["subject"].nunique()))
    models = build_models(feature_sets)
    scores = {name: {"auc": [], "ap": []} for name, _ in models}
    scores["organ-rate baseline"] = {"auc": [], "ap": []}
    scores["always-accept"] = {"auc": [], "ap": []}
    oof = {name: np.zeros(len(df)) for name in scores}

    for tr, te in cv.split(df, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        Xtr, Xte = X.iloc[tr], X.iloc[te]

        p = organ_baseline(tr_df, te_df, y[tr].mean())
        scores["organ-rate baseline"]["auc"].append(roc_auc_score(y[te], p))
        scores["organ-rate baseline"]["ap"].append(average_precision_score(y[te], p))
        oof["organ-rate baseline"][te] = p

        p = np.full(len(te), y[tr].mean())
        scores["always-accept"]["auc"].append(0.5)
        scores["always-accept"]["ap"].append(average_precision_score(y[te], p))
        oof["always-accept"][te] = p

        for name, pipe in models:
            pipe.fit(Xtr, y[tr])
            p = pipe.predict_proba(Xte)[:, 1]
            scores[name]["auc"].append(roc_auc_score(y[te], p))
            scores[name]["ap"].append(average_precision_score(y[te], p))
            oof[name][te] = p

    rej_ap = {n: average_precision_score(1 - y, 1 - oof[n]) for n in scores if scores[n]["ap"]}

    print(f"\n{'model':32s} {'ROC-AUC':>16s} {'PR-AUC(acc)':>16s} {'PR-AUC(rej)':>12s}")
    print("-" * 80)
    ranked = sorted(scores.items(), key=lambda kv: -np.mean(kv[1]["ap"] or [0]))
    for name, s in ranked:
        if not s["ap"]:
            continue
        print(f"{name:32s} {np.mean(s['auc']):>8.3f} +/-{np.std(s['auc']):<6.3f} "
              f"{np.mean(s['ap']):>8.3f} +/-{np.std(s['ap']):<6.3f} {rej_ap[name]:>12.3f}")
    print(f"\n(random-model PR-AUC: accept {y.mean():.3f}, reject {1-y.mean():.3f})")

    best = ranked[0][0]
    print(f"\n=== operating points: {best} ===")
    print(f"{'cutoff':>8s} {'auto-acc%':>10s} {'bad slipping':>13s} "
          f"{'bad caught':>11s} {'review%':>9s} {'review purity':>14s}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        acc = oof[best] >= t
        n_acc = acc.sum()
        slip = (y[acc] == 0).mean() if n_acc else float("nan")     # bad masks waved through
        caught = ((~acc) & (y == 0)).sum() / max((y == 0).sum(), 1)  # share of all bad caught
        purity = (y[~acc] == 0).mean() if (~acc).any() else float("nan")  # review pile that is real
        print(f"{t:>8.2f} {100*n_acc/len(y):>9.1f}% {100*slip:>12.1f}% "
              f"{100*caught:>10.1f}% {100*(1-n_acc/len(y)):>8.1f}% {100*purity:>13.1f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = df[["subject", "organ", "iou", "label"]].copy()
    for name in oof:
        out[name.replace(" ", "_")] = oof[name]
    out.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)
    print(f"\nout-of-fold predictions -> {os.path.join(OUT_DIR, 'oof_predictions.csv')}")
    return scores


def leave_one_organ_out(df, feature_sets, min_n):
    """Train on every organ EXCEPT one, test on the held-out organ.

    This is the test of whether the quality signal is structural or memorised. Raw
    features cannot transfer, because the model has no idea what scale an unseen
    structure operates at. Organ-relative z-scores should, because "3 SD below typical"
    means the same thing for a pancreas as for a liver.
    """
    usable = []
    for o, g in df.groupby("organ"):
        n_pos, n_neg = int(g["label"].sum()), int((1 - g["label"]).sum())
        if len(g) >= min_n and min(n_pos, n_neg) >= LOO_MIN_MINORITY:
            usable.append((o, g))
    print(f"\n=== leave-one-organ-out ({len(usable)} organs, >= {min_n} rows each) ===")
    print("train on all OTHER organs, test on the held-out one\n")

    sets = {k: v for k, v in feature_sets.items() if not v[1]}  # no organ one-hot in LOO
    results = {k: [] for k in sets}
    per_organ = {}

    for organ, test_df in usable:
        train_df = df[df["organ"] != organ]
        row = {}
        for set_name, (cols, _) in sets.items():
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("clf", RandomForestClassifier(n_estimators=300,
                                     min_samples_leaf=2, class_weight="balanced_subsample",
                                     random_state=SEED, n_jobs=-1))])
            pipe.fit(train_df[cols], train_df["label"])
            p = pipe.predict_proba(test_df[cols])[:, 1]
            auc = roc_auc_score(test_df["label"], p)
            results[set_name].append(auc)
            row[set_name] = auc
        per_organ[organ] = (len(test_df), test_df["label"].mean(), row)

    header = f"{'organ':30s} {'n':>6s} {'acc%':>6s} " + " ".join(f"{k:>10s}" for k in sets)
    print(header); print("-" * len(header))
    for organ in sorted(per_organ, key=lambda o: -per_organ[o][0]):
        n, rate, row = per_organ[organ]
        n_min = int(min(test_df["label"].sum(), (1 - test_df["label"]).sum()))
        print(f"{organ:30s} {n:>6d} {100*rate:>5.0f}% {n_min:>6d} " +
              " ".join(f"{row[k]:>10.3f}" for k in sets))
    print("-" * len(header))
    print(f"{'MEAN':30s} {'':>6s} {'':>6s} " +
          " ".join(f"{np.mean(results[k]):>10.3f}" for k in sets))

    out = pd.DataFrame([{"organ": o, "n": per_organ[o][0],
                         "accept_rate": per_organ[o][1], **per_organ[o][2]}
                        for o in per_organ])
    out.to_csv(os.path.join(OUT_DIR, "loo_by_organ.csv"), index=False)
    print(f"\nper-organ LOO results -> {os.path.join(OUT_DIR, 'loo_by_organ.csv')}")


def importances(df, feature_sets):
    cols = feature_sets.get("raw+rel", feature_sets["raw"])[0]
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                    class_weight="balanced_subsample",
                                                    random_state=SEED, n_jobs=-1))])
    pipe.fit(df[cols], df["label"])
    imp = sorted(zip(cols, pipe.named_steps["clf"].feature_importances_),
                 key=lambda t: -t[1])
    print("\n=== feature importance (all features, whole dataset) ===")
    for name, v in imp[:15]:
        print(f"  {name:32s} {v:.4f}")


def main():
    df, raw_cols, rel_cols = load(CSV, IOU_ACCEPT, DROP_EMPTY)
    feature_sets = make_feature_sets(raw_cols, rel_cols)
    print("feature sets: " + ", ".join(f"{k} ({len(v[0])}{'+organ' if v[1] else ''})"
                                       for k, v in feature_sets.items()))
    evaluate(df, feature_sets)
    if RUN_LOO and rel_cols:
        leave_one_organ_out(df, feature_sets, LOO_MIN_N)
    importances(df, feature_sets)


if __name__ == "__main__":
    main()