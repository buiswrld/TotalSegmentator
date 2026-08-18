"""
Optional analysis stage for the QC pipeline: an ablation suite for the mask-quality
classifier, run against a curated dataset (stage 4 / curate_dataset.py output).

Four experiments, each independently runnable via --skip-* flags:

  1. FEATURE ABLATION      which feature groups carry the signal
  2. THRESHOLD SWEEP       is the IoU 0.90 cutoff load-bearing
  3. WITHIN-ORGAN AUC      does the model discriminate WITHIN a structure, or only
                           rank easy structures above hard ones
  4. LEAVE-ONE-FAMILY-OUT  honest transfer test: holding out one rib is meaningless
                           when 23 other ribs stay in training

The central question these are built to answer: size features rank high in the
importance table, but IoU is size-biased by construction (for a sphere of radius r
with boundary error d, IoU ~= 1 - 3d/r), so a fixed IoU threshold is a much tighter
physical standard for small structures. Size may therefore predict the LABEL without
predicting mask QUALITY. Experiments 1 and 3 together test that: if size matters only
via the metric artifact, dropping it should hurt pooled AUC far more than within-organ AUC.

Reads a curated dataset CSV from stage 4 directly (labels, empty-row dropping, and
organ-relative z-score joining are already done there - this script never re-derives
them). See totalsegmentator/qc_columns.py for the full column-name glossary.

Writes, all under --output-dir:
  feature_ablation.csv   Ablation Configuration Label/Kind (config/kind),
                         Number of Features Used (n_features), AUC,
                         PR-AUC for Reject Class (pr_auc_reject),
                         Mean Within-Group AUC (within_organ_auc),
                         Number of Groups in Within-Group AUC (n_organs_within),
                         Pooled/Within-Group AUC Change vs Full Model (delta_auc/delta_within)
  threshold_sweep.csv    IoU Accept Threshold Tested (iou_threshold), Accept Rate,
                         AUC, pr_auc_reject, within_organ_auc, n_organs_within
  threshold_sweep_by_family.csv   one row per (threshold, family): iou_threshold,
                         Anatomical Family Grouping (family), Number of Masks (n),
                         Accept Rate, AUC (NaN if that family lacked enough data AT
                         THAT THRESHOLD), Sufficient Data for AUC (has_sufficient_data)
  threshold_sweep_family_volatility.csv   one row per family, summarizing the sweep
                         above: Number of Thresholds with Sufficient Data
                         (n_thresholds_evaluated), AUC min/max/range across those
                         thresholds, Accept Rate min/max/range across ALL thresholds
                         (no eligibility gate) - sorted by AUC range descending, so the
                         top rows directly answer "which families are most
                         threshold-sensitive" rather than requiring you to eyeball
                         threshold_sweep_by_family.csv yourself
  within_organ_auc.csv   Organ/Structure Name, Number of Masks (n), Accept Rate, AUC,
                         Anatomical Family Grouping (family)
  within_family_auc.csv  Anatomical Family Grouping (family), Number of Masks (n),
                         Accept Rate, AUC
  family_loo.csv         family, Number of Masks (n), Number of Distinct Organs
                         (n_organs), Accept Rate, AUC
  organ_loo.csv          same columns, held out one organ at a time instead of family

Run:  python experiments/pipeline/ablations.py --dataset-csv datasets/classifier_train.csv --output-dir results/ablations
"""

import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.train import resolve_feature_columns
from experiments.pipeline.common import write_manifest
from totalsegmentator.qc_columns import (
    COL_SUBJECT, COL_ORGAN, COL_IOU, COL_TRAINING_LABEL,
    COL_NUM_VOXELS, COL_VOLUME_MM3, COL_BBOX_VOLUME_MM3,
    COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL,
    COL_BBOX_X_REL, COL_BBOX_Y_REL, COL_BBOX_Z_REL, COL_MASK_TO_BBOX_RATIO,
    COL_NUM_COMPONENTS, COL_LARGEST_COMPONENT_FRACTION,
    COL_TOUCHES_BOUNDARY, COL_BOUNDARY_FRACTION,
    COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU,
    COL_ANATOMICAL_FAMILY, COL_N_ORGANS, COL_ABLATION_CONFIG, COL_ABLATION_KIND,
    COL_N_FEATURES, COL_AUC, COL_PR_AUC_REJECT, COL_WITHIN_ORGAN_AUC,
    COL_N_GROUPS_WITHIN, COL_DELTA_AUC, COL_DELTA_WITHIN_AUC, COL_IOU_THRESHOLD,
    COL_N_MASKS, COL_ACCEPT_RATE,
    COL_SUFFICIENT_DATA_FLAG, COL_AUC_MIN, COL_AUC_MAX, COL_AUC_RANGE,
    COL_ACCEPT_RATE_MIN, COL_ACCEPT_RATE_MAX, COL_ACCEPT_RATE_RANGE,
    COL_N_THRESHOLDS_EVALUATED,
)

FEATURE_GROUPS = {
    "size": [COL_NUM_VOXELS, COL_VOLUME_MM3, COL_BBOX_VOLUME_MM3],
    "position": [COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL],
    "extent": [COL_BBOX_X_REL, COL_BBOX_Y_REL, COL_BBOX_Z_REL, COL_MASK_TO_BBOX_RATIO],
    "component": [COL_NUM_COMPONENTS, COL_LARGEST_COMPONENT_FRACTION],
    "boundary": [COL_TOUCHES_BOUNDARY, COL_BOUNDARY_FRACTION],
    "intensity": [COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU],
}

# Ordered: first match wins, so specific patterns must precede general ones.
FAMILY_PATTERNS = [
    ("rib",              r"^rib_"),
    ("vertebrae",        r"^vertebrae"),
    ("costal_cartilage", r"costal_cartilage"),
    ("lung",             r"^lung"),
    ("heart",            r"^heart|^atrial|^ventricle|^myocardium|^pulmonary_artery"),
    ("great_vessel",     r"aorta|vena_cava|brachiocephalic|subclavian|carotid|"
                         r"pulmonary_vein|portal_vein|iliac_arter|iliac_vena"),
    ("digestive",        r"^esophagus|^stomach|^duodenum|^small_bowel|^colon"),
    ("solid_abdominal",  r"^liver|^spleen|^pancreas|^gallbladder"),
    ("urinary",          r"^kidney|^urinary_bladder|^adrenal_gland"),
    ("muscle",           r"^autochthon|^iliopsoas|^gluteus"),
    ("pelvic_bone",      r"^hip|^sacrum|^femur"),
    ("shoulder_bone",    r"^scapula|^clavicula|^humerus|^sternum"),
    ("neuro",            r"^brain|^spinal_cord|^skull"),
    ("airway_neck",      r"^trachea|^thyroid"),
    ("reproductive",     r"^prostate"),
]


def family_of(organ):
    for name, pat in FAMILY_PATTERNS:
        if re.search(pat, organ):
            return name
    return "other"


def make_model(n_trees, seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=n_trees,
                             min_samples_leaf=2, class_weight="balanced_subsample",
                             random_state=seed, n_jobs=-1))])


def oof_predict(df, cols, y, groups, n_splits, n_trees, seed):
    """Subject-grouped out-of-fold probabilities."""
    p = np.zeros(len(y))
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in cv.split(df, y, groups):
        m = make_model(n_trees, seed)
        m.fit(df.iloc[tr][cols], y[tr])
        p[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
    return p


def feature_importance(df, cols, y, n_trees, seed):
    """(feature, importance) sorted desc, from one RF fit on the whole dataset.

    Diagnostic only (not cross-validated) - used to pick which single features are
    worth an individual ablation. Column order is preserved through the median imputer,
    so importances line up with `cols`.
    """
    m = make_model(n_trees, seed)
    m.fit(df[cols], y)
    imp = m.named_steps["clf"].feature_importances_
    return sorted(zip(cols, imp), key=lambda kv: -kv[1])


def within_organ_auc(df, p, y, key, min_n, min_minority):
    """Mean AUC computed SEPARATELY inside each group (organ or family).

    Pooled AUC rewards ranking easy structures above hard ones. Within-group AUC
    strips that out: it only asks whether, among masks of the SAME structure, the
    good ones score higher. A large pooled-minus-within gap means most of the
    apparent performance is organ difficulty, not mask quality.
    """
    out = []
    for name, g in df.assign(_p=p, _y=y).groupby(key):
        n_pos, n_neg = int(g._y.sum()), int((1 - g._y).sum())
        if len(g) >= min_n and min(n_pos, n_neg) >= min_minority:
            out.append({key: name, COL_N_MASKS: len(g), COL_ACCEPT_RATE: g._y.mean(),
                        COL_AUC: roc_auc_score(g._y, g._p)})
    return pd.DataFrame(out)


def family_breakdown_at_threshold(df, p, y, min_n, min_minority):
    """Per-family (n, accept rate, AUC) for one threshold's predictions.

    Unlike within_organ_auc(), never silently drops a family: mask count and accept
    rate are always reported, and AUC is NaN with has_sufficient_data=False when the
    family doesn't meet min_n/min_minority AT THIS THRESHOLD specifically (eligibility
    can hold at a loose threshold and fail at a strict one as the accept rate
    collapses) - so a family becoming impossible to evaluate at a strict cutoff is
    visible in the output instead of quietly vanishing.
    """
    rows = []
    for name, g in df.assign(_p=p, _y=y).groupby(COL_ANATOMICAL_FAMILY):
        n_pos, n_neg = int(g._y.sum()), int((1 - g._y).sum())
        sufficient = len(g) >= min_n and min(n_pos, n_neg) >= min_minority
        rows.append({COL_ANATOMICAL_FAMILY: name, COL_N_MASKS: len(g),
                     COL_ACCEPT_RATE: g._y.mean(),
                     COL_AUC: roc_auc_score(g._y, g._p) if sufficient else np.nan,
                     COL_SUFFICIENT_DATA_FLAG: sufficient})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiments

def experiment_ablation(df, all_cols, y, groups, n_splits, n_trees, seed,
                        within_min_n, within_min_minority, output_dir):
    print("\n" + "=" * 74)
    print("1. FEATURE ABLATION")
    print("=" * 74)
    groups_present = {g: [c for c in cs if c in all_cols]
                      for g, cs in FEATURE_GROUPS.items()}
    groups_present = {g: cs for g, cs in groups_present.items() if cs}

    rows = []

    def run(label, cols, kind):
        if not cols:
            return
        p = oof_predict(df, cols, y, groups, n_splits, n_trees, seed)
        w = within_organ_auc(df, p, y, COL_ORGAN, within_min_n, within_min_minority)
        rows.append({COL_ABLATION_CONFIG: label, COL_ABLATION_KIND: kind, COL_N_FEATURES: len(cols),
                     COL_AUC: roc_auc_score(y, p),
                     COL_PR_AUC_REJECT: average_precision_score(1 - y, 1 - p),
                     COL_WITHIN_ORGAN_AUC: w[COL_AUC].mean() if len(w) else np.nan,
                     COL_N_GROUPS_WITHIN: len(w)})
        r = rows[-1]
        print(f"  {label:28s} feats={r[COL_N_FEATURES]:>3d}  AUC={r[COL_AUC]:.3f}  "
              f"rej={r[COL_PR_AUC_REJECT]:.3f}  within-organ={r[COL_WITHIN_ORGAN_AUC]:.3f}")

    print("\nfull model:")
    run("ALL", all_cols, "full")

    print("\nleave-one-group-out (how much is lost without this group):")
    for g, cs in groups_present.items():
        run(f"without {g}", [c for c in all_cols if c not in cs], "drop_group")

    print("\nonly-one-group (how much this group carries alone):")
    for g, cs in groups_present.items():
        run(f"only {g}", cs, "only_group")

    # ---- single-feature ablation for the top features by importance ----
    ranked_imp = feature_importance(df, all_cols, y, n_trees, seed)
    top_feats = [f for f, _ in ranked_imp[:6]]
    print("\ntop 6 features by RF importance (whole-dataset fit, diagnostic):")
    for f, v in ranked_imp[:6]:
        print(f"  {v:.4f}  {f}")

    print("\ndrop-one-feature (marginal loss when this single feature is removed):")
    for f in top_feats:
        run(f"without {f}", [c for c in all_cols if c != f], "drop_feature")

    print("\nonly-one-feature (how much this single feature carries alone):")
    for f in top_feats:
        run(f"only {f}", [f], "only_feature")

    res = pd.DataFrame(rows)
    full = res[res[COL_ABLATION_KIND] == "full"].iloc[0]
    res[COL_DELTA_AUC] = res[COL_AUC] - full[COL_AUC]
    res[COL_DELTA_WITHIN_AUC] = res[COL_WITHIN_ORGAN_AUC] - full[COL_WITHIN_ORGAN_AUC]
    res.to_csv(os.path.join(output_dir, "feature_ablation.csv"), index=False)

    # ---- unified table: every ablation, pooled + within-organ AUC and both deltas,
    #      sorted by pooled delta (most harmful first) ----
    print("\n  --- all ablations, sorted by pooled delta vs full model ---")
    print(f"  {'config':26s} {'pooledAUC':>9s} {'withinAUC':>9s} {'d_pooled':>9s} {'d_within':>9s}")
    print("  " + "-" * 66)
    ordered = res[res[COL_ABLATION_KIND] != "full"].sort_values(COL_DELTA_AUC)
    for _, r in ordered.iterrows():
        print(f"  {r[COL_ABLATION_CONFIG]:26s} {r[COL_AUC]:>9.3f} {r[COL_WITHIN_ORGAN_AUC]:>9.3f} "
              f"{r[COL_DELTA_AUC]:>+9.3f} {r[COL_DELTA_WITHIN_AUC]:>+9.3f}")

    # ---- interpretation: which groups separate easy/hard structures rather than
    #      judging mask quality (hurt pooled much more than within-organ when dropped) ----
    print("\n  --- interpretation (dropped groups) ---")
    drops = res[res[COL_ABLATION_KIND] == "drop_group"].sort_values(COL_DELTA_AUC)
    flagged = []
    for _, r in drops.iterrows():
        g = r[COL_ABLATION_CONFIG].replace("without ", "")
        # more-negative delta_auc than delta_within => removing it costs pooled ranking
        # more than within-organ quality => the group was mostly an easy/hard separator.
        easy_hard_gap = r[COL_DELTA_WITHIN_AUC] - r[COL_DELTA_AUC]
        tag = "  <- separates easy/hard, not good/bad" if easy_hard_gap > 0.02 else ""
        if easy_hard_gap > 0.02:
            flagged.append(g)
        print(f"  dropping {g:12s} pooled {r[COL_DELTA_AUC]:+.3f}   within-organ {r[COL_DELTA_WITHIN_AUC]:+.3f}{tag}")
    print("\n  A group that hurts POOLED far more than WITHIN-ORGAN is mostly")
    print("  separating easy structures from hard ones, not good masks from bad.")
    if flagged:
        print(f"  flagged (pooled-dominant): {', '.join(flagged)}")
    else:
        print("  none flagged: every group that matters contributes to WITHIN-organ")
        print("  discrimination too, i.e. real mask-quality signal, not just easy/hard ranking.")
    return res


def experiment_threshold(df, all_cols, groups, thresholds, n_splits, n_trees, seed,
                         within_min_n, within_min_minority, family_min_n, output_dir):
    print("\n" + "=" * 74)
    print("2. THRESHOLD SWEEP")
    print("=" * 74)
    rows = []
    family_rows = []
    print(f"\n{'IoU cut':>8s} {'accept%':>9s} {'AUC':>7s} {'rej PR':>8s} "
          f"{'within-organ':>13s} {'n organs':>9s}")
    for t in thresholds:
        y = (df[COL_IOU] >= t).astype(int).to_numpy()
        if y.mean() in (0.0, 1.0):
            continue
        # One oof_predict() per threshold, reused for both the pooled/within-organ row
        # below and the per-family breakdown - avoids re-fitting models a second time
        # for the same threshold.
        p = oof_predict(df, all_cols, y, groups, n_splits, n_trees, seed)
        w = within_organ_auc(df, p, y, COL_ORGAN, within_min_n, within_min_minority)
        rows.append({COL_IOU_THRESHOLD: t, COL_ACCEPT_RATE: y.mean(),
                     COL_AUC: roc_auc_score(y, p),
                     COL_PR_AUC_REJECT: average_precision_score(1 - y, 1 - p),
                     COL_WITHIN_ORGAN_AUC: w[COL_AUC].mean() if len(w) else np.nan,
                     COL_N_GROUPS_WITHIN: len(w)})
        r = rows[-1]
        print(f"{t:>8.2f} {100*r[COL_ACCEPT_RATE]:>8.1f}% {r[COL_AUC]:>7.3f} "
              f"{r[COL_PR_AUC_REJECT]:>8.3f} {r[COL_WITHIN_ORGAN_AUC]:>13.3f} "
              f"{r[COL_N_GROUPS_WITHIN]:>9d}")

        fam = family_breakdown_at_threshold(df, p, y, family_min_n, within_min_minority)
        fam.insert(0, COL_IOU_THRESHOLD, t)
        family_rows.append(fam)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(output_dir, "threshold_sweep.csv"), index=False)
    print("\n  Stable AUC across thresholds means 0.90 is not load-bearing.")
    print("  Note accept% shifts a lot, so PR-AUC moves with the class balance;")
    print("  compare AUC (no prevalence floor) across rows, not PR-AUC.")

    # ---- per-family breakdown: does the pooled/within-organ stability above hold
    #      for every family individually, or is it hiding families moving in opposite
    #      directions as the threshold changes? ----
    fam_df = pd.concat(family_rows, ignore_index=True)
    fam_df.to_csv(os.path.join(output_dir, "threshold_sweep_by_family.csv"), index=False)

    volatility = []
    for name, g in fam_df.groupby(COL_ANATOMICAL_FAMILY):
        evaluable = g[g[COL_SUFFICIENT_DATA_FLAG]]
        volatility.append({
            COL_ANATOMICAL_FAMILY: name,
            COL_N_THRESHOLDS_EVALUATED: len(evaluable),
            COL_AUC_MIN: evaluable[COL_AUC].min() if len(evaluable) else np.nan,
            COL_AUC_MAX: evaluable[COL_AUC].max() if len(evaluable) else np.nan,
            COL_AUC_RANGE: (evaluable[COL_AUC].max() - evaluable[COL_AUC].min()) if len(evaluable) else np.nan,
            COL_ACCEPT_RATE_MIN: g[COL_ACCEPT_RATE].min(),
            COL_ACCEPT_RATE_MAX: g[COL_ACCEPT_RATE].max(),
            COL_ACCEPT_RATE_RANGE: g[COL_ACCEPT_RATE].max() - g[COL_ACCEPT_RATE].min(),
        })
    vol_df = pd.DataFrame(volatility).sort_values(COL_AUC_RANGE, ascending=False)
    vol_df.to_csv(os.path.join(output_dir, "threshold_sweep_family_volatility.csv"), index=False)

    print("\n  --- per-family AUC volatility across the threshold sweep ---")
    print("  (large AUC range despite pooled stability = this family IS threshold-sensitive)")
    print(f"  {'family':20s} {'n_thr':>6s} {'AUC min':>8s} {'AUC max':>8s} {'AUC rng':>8s} "
          f"{'acc% rng':>9s}")
    for _, r in vol_df.iterrows():
        auc_range = f"{r[COL_AUC_RANGE]:.3f}" if pd.notna(r[COL_AUC_RANGE]) else "n/a"
        auc_min = f"{r[COL_AUC_MIN]:.3f}" if pd.notna(r[COL_AUC_MIN]) else "n/a"
        auc_max = f"{r[COL_AUC_MAX]:.3f}" if pd.notna(r[COL_AUC_MAX]) else "n/a"
        print(f"  {r[COL_ANATOMICAL_FAMILY]:20s} {int(r[COL_N_THRESHOLDS_EVALUATED]):>6d} "
              f"{auc_min:>8s} {auc_max:>8s} {auc_range:>8s} "
              f"{100*r[COL_ACCEPT_RATE_RANGE]:>8.1f}%")
    insufficient = fam_df[~fam_df[COL_SUFFICIENT_DATA_FLAG]]
    if len(insufficient):
        print(f"\n  {insufficient[COL_ANATOMICAL_FAMILY].nunique()} family/families had at least one "
              f"threshold with insufficient data to compute an AUC (see has_sufficient_data=False "
              f"rows in threshold_sweep_by_family.csv) - typically at the strictest cutoffs, where "
              f"too few accepts remain.")
    return res


def experiment_within(df, all_cols, y, groups, n_splits, n_trees, seed,
                      within_min_n, within_min_minority, family_min_n, output_dir):
    print("\n" + "=" * 74)
    print("3. WITHIN-ORGAN AND WITHIN-FAMILY AUC")
    print("=" * 74)
    p = oof_predict(df, all_cols, y, groups, n_splits, n_trees, seed)
    pooled = roc_auc_score(y, p)

    w_org = within_organ_auc(df, p, y, COL_ORGAN, within_min_n, within_min_minority)
    w_fam = within_organ_auc(df, p, y, COL_ANATOMICAL_FAMILY, family_min_n, within_min_minority)

    print(f"\n  pooled AUC              {pooled:.3f}")
    print(f"  mean within-ORGAN AUC   {w_org[COL_AUC].mean():.3f}   ({len(w_org)} organs)")
    print(f"  mean within-FAMILY AUC  {w_fam[COL_AUC].mean():.3f}   ({len(w_fam)} families)")
    print(f"  pooled - within-organ   {pooled - w_org[COL_AUC].mean():+.3f}")
    print("\n  The gap is the share of pooled AUC that comes from ranking structures")
    print("  against each other rather than judging masks of the same structure.")

    w_org = w_org.assign(**{COL_ANATOMICAL_FAMILY: [family_of(o) for o in w_org[COL_ORGAN]]})
    w_org.sort_values(COL_AUC).to_csv(os.path.join(output_dir, "within_organ_auc.csv"), index=False)
    w_fam.to_csv(os.path.join(output_dir, "within_family_auc.csv"), index=False)

    display_cols = [COL_ORGAN, COL_N_MASKS, COL_ACCEPT_RATE, COL_AUC]
    print("\n  weakest organs (model barely discriminates within them):")
    print(w_org.nsmallest(6, COL_AUC)[display_cols].to_string(index=False))
    print("\n  strongest:")
    print(w_org.nlargest(6, COL_AUC)[display_cols].to_string(index=False))
    print("\n  by family:")
    print(w_fam.sort_values(COL_AUC).to_string(index=False))
    return w_org, w_fam


def leave_one_key_out(df, all_cols, key, min_n, min_minority, n_trees, seed):
    """Train on every level of `key` except one, test on the held-out level.

    Same model/feature set regardless of key, so organ-vs-family results are directly
    comparable and any difference is purely the granularity of what was held out.
    Returns a DataFrame with one row per held-out level.
    """
    groups_ = [(k, g) for k, g in df.groupby(key)
               if len(g) >= min_n and g[COL_TRAINING_LABEL].nunique() == 2
               and min(g[COL_TRAINING_LABEL].sum(), (1 - g[COL_TRAINING_LABEL]).sum()) >= min_minority]
    rows = []
    for k, test in groups_:
        train = df[df[key] != k]
        m = make_model(n_trees, seed)
        m.fit(train[all_cols], train[COL_TRAINING_LABEL])
        p = m.predict_proba(test[all_cols])[:, 1]
        rows.append({key: k, COL_N_MASKS: len(test), COL_N_ORGANS: test[COL_ORGAN].nunique(),
                     COL_ACCEPT_RATE: test[COL_TRAINING_LABEL].mean(),
                     COL_AUC: roc_auc_score(test[COL_TRAINING_LABEL], p)})
    return pd.DataFrame(rows)


def experiment_family_loo(df, all_cols, y, groups, family_min_n, within_min_n,
                          within_min_minority, n_trees, seed, output_dir):
    print("\n" + "=" * 74)
    print("4. LEAVE-ONE-FAMILY-OUT")
    print("=" * 74)
    print("\n  Holding out ONE rib leaves 23 near-identical ribs in training, so")
    print("  per-organ LOO overstates transfer. This holds out the whole family.")

    res = leave_one_key_out(df, all_cols, COL_ANATOMICAL_FAMILY, family_min_n, within_min_minority, n_trees, seed)
    if res.empty:
        print("  no families with enough data")
        return None

    res = res.sort_values(COL_AUC)
    print(f"\n{'family':20s} {'n':>6s} {'organs':>7s} {'accept%':>8s} {'AUC':>7s}")
    print("-" * 52)
    for _, r in res.iterrows():
        print(f"{r[COL_ANATOMICAL_FAMILY]:20s} {r[COL_N_MASKS]:>6d} {r[COL_N_ORGANS]:>7d} "
              f"{100*r[COL_ACCEPT_RATE]:>7.1f}% {r[COL_AUC]:>7.3f}")
    print("-" * 52)
    print(f"{'MEAN':20s} {'':>6s} {'':>7s} {'':>8s} {res[COL_AUC].mean():>7.3f}")
    print(f"{'MEDIAN':20s} {'':>6s} {'':>7s} {'':>8s} {res[COL_AUC].median():>7.3f}")
    print(f"\n  families above chance: {(res[COL_AUC] > 0.5).sum()}/{len(res)}")
    res.to_csv(os.path.join(output_dir, "family_loo.csv"), index=False)

    # ---- leave-one-ORGAN-out under the identical model/features, for a fair contrast ----
    org = leave_one_key_out(df, all_cols, COL_ORGAN, within_min_n, within_min_minority, n_trees, seed)
    org.to_csv(os.path.join(output_dir, "organ_loo.csv"), index=False)
    print("\n  --- organ-LOO vs family-LOO (same model & features) ---")
    print(f"  leave-one-ORGAN-out    mean {org[COL_AUC].mean():.3f}   median {org[COL_AUC].median():.3f}   ({len(org)} organs)")
    print(f"  leave-one-FAMILY-out   mean {res[COL_AUC].mean():.3f}   median {res[COL_AUC].median():.3f}   ({len(res)} families)")
    print(f"  drop (organ - family)  mean {org[COL_AUC].mean() - res[COL_AUC].mean():+.3f}")
    print("\n  Organ-LOO is optimistic: holding out one rib leaves ~23 near-identical")
    print("  ribs in training (22 of the usable organs are ribs), so the model still")
    print("  sees the structure type. Family-LOO removes the whole type at once - the")
    print("  drop is the price of true novel-structure transfer, not a bug.")
    return res


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-csv", required=True,
                        help="Curated dataset CSV from stage 4 (curate_dataset.py), e.g. classifier_train.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=200,
                        help="Lower than the 400 used for headline train.py numbers; ablations run "
                             "many fits and the ranking is stable at 200 (default: 200).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-relative-features", action="store_true",
                        help="Include the organ-relative _z features already joined by curate_dataset.py "
                             "(default: off - z-scores add little on organs seen during training; "
                             "turn on to test family-transfer specifically).")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                        help="IoU accept thresholds to sweep in experiment 2.")
    parser.add_argument("--within-min-n", type=int, default=25, help="Rows needed for a per-organ AUC.")
    parser.add_argument("--within-min-minority", type=int, default=5, help="Rows of the rarer class needed.")
    parser.add_argument("--family-min-n", type=int, default=40, help="Rows needed for a per-family AUC.")
    parser.add_argument("--skip-feature-ablation", action="store_true")
    parser.add_argument("--skip-threshold-sweep", action="store_true")
    parser.add_argument("--skip-within-group", action="store_true")
    parser.add_argument("--skip-family-loo", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.dataset_csv)
    df[COL_ANATOMICAL_FAMILY] = [family_of(o) for o in df[COL_ORGAN]]

    raw_cols, rel_cols = resolve_feature_columns(args.dataset_csv)
    all_cols = raw_cols + (rel_cols if args.use_relative_features else [])
    # zero-variance columns (e.g. is_empty, constant once empty rows are already
    # dropped by stage 4) carry no signal and just slow the fits down - drop them.
    all_cols = [c for c in all_cols if df[c].nunique(dropna=False) > 1]

    y = df[COL_TRAINING_LABEL].to_numpy()
    groups = df[COL_SUBJECT].to_numpy()

    print(f"{len(df)} masks | {df[COL_SUBJECT].nunique()} subjects | "
          f"{df[COL_ORGAN].nunique()} organs | {df[COL_ANATOMICAL_FAMILY].nunique()} families | "
          f"{100*y.mean():.1f}% accept")
    print(f"{len(all_cols)} features ({'raw+relative' if args.use_relative_features else 'raw only'})\n")
    fam_counts = df[COL_ANATOMICAL_FAMILY].value_counts()
    print("family sizes:")
    for f, n in fam_counts.items():
        print(f"  {f:20s} {n:>5d}  ({df[df[COL_ANATOMICAL_FAMILY]==f][COL_ORGAN].nunique()} organs)")
    if "other" in fam_counts:
        unmapped = sorted(df[df[COL_ANATOMICAL_FAMILY] == "other"][COL_ORGAN].unique())[:15]
        print(f"\n  unmapped -> 'other': {unmapped}")
        print("  (extend FAMILY_PATTERNS if any of these should be grouped)")

    if not args.skip_feature_ablation:
        experiment_ablation(df, all_cols, y, groups, args.n_splits, args.n_trees, args.seed,
                            args.within_min_n, args.within_min_minority, args.output_dir)
    if not args.skip_threshold_sweep:
        experiment_threshold(df, all_cols, groups, args.thresholds, args.n_splits, args.n_trees, args.seed,
                             args.within_min_n, args.within_min_minority, args.family_min_n, args.output_dir)
    if not args.skip_within_group:
        experiment_within(df, all_cols, y, groups, args.n_splits, args.n_trees, args.seed,
                          args.within_min_n, args.within_min_minority, args.family_min_n, args.output_dir)
    if not args.skip_family_loo:
        experiment_family_loo(df, all_cols, y, groups, args.family_min_n, args.within_min_n,
                              args.within_min_minority, args.n_trees, args.seed, args.output_dir)

    write_manifest(
        os.path.join(args.output_dir, "ablations_manifest.json"),
        stage="ablations", dataset_csv=args.dataset_csv, n_splits=args.n_splits,
        n_trees=args.n_trees, seed=args.seed, use_relative_features=args.use_relative_features,
        thresholds=args.thresholds, n_rows=len(df), n_subjects=int(df[COL_SUBJECT].nunique()),
        n_features=len(all_cols),
    )
    print(f"\nall CSVs written to {args.output_dir}")


if __name__ == "__main__":
    main()
