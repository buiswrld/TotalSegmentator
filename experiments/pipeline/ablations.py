"""
Ablation suite for the mask quality classifier.

Four experiments, each independently runnable via the RUN_* flags:

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

Run:  python ablations.py
"""

import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import os, re, sys, csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from build_reference_table import load_reference, add_relative_features
    HAVE_REF = True
except Exception:
    HAVE_REF = False

# ============================ CONFIG ============================
CSV           = r"C:\Users\ansar\Algoverse\report_ct_val\combined_metrics.csv"
REFERENCE_CSV = r"C:\Users\ansar\Algoverse\report_ct\reference_stats.csv"
OUT_DIR       = r"C:\Users\ansar\Algoverse\report_ct_val\ablations"

IOU_ACCEPT = 0.90
N_SPLITS   = 5
N_TREES    = 200          # lower than the 400 used for headline numbers; ablations run
                          # many fits and the ranking is stable at 200
USE_REFERENCE = False     # z-scores add little on seen organs; turn on for family transfer
SEED = 0

RUN_ABLATION   = True
RUN_THRESHOLD  = True
RUN_WITHIN     = True
RUN_FAMILY_LOO = True

THRESHOLDS   = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
WITHIN_MIN_N = 25         # rows needed for a per-organ AUC
WITHIN_MIN_MINORITY = 5   # rows of the rarer class needed
FAMILY_MIN_N = 40
# ================================================================

LEAK_COLS = ["dice", "iou", "tp", "fp", "fn", "gt_vox", "status", "accept"]
ID_COLS   = ["subject", "organ", "orig_axcodes", "pred_vox", "family", "label"]

FEATURE_GROUPS = {
    "size":      ["num_voxels", "volume_mm3", "bbox_volume_mm3"],
    "position":  ["centroid_x_rel", "centroid_y_rel", "centroid_z_rel"],
    "extent":    ["bbox_x_rel", "bbox_y_rel", "bbox_z_rel", "mask_to_bbox_ratio"],
    "component": ["num_components", "largest_component_fraction"],
    "boundary":  ["touches_boundary", "boundary_fraction"],
    "intensity": ["mean_HU", "median_HU", "std_HU", "p05_HU", "p95_HU"],
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


def make_model():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=N_TREES,
                             min_samples_leaf=2, class_weight="balanced_subsample",
                             random_state=SEED, n_jobs=-1))])


def oof_predict(df, cols, y, groups):
    """Subject-grouped out-of-fold probabilities."""
    p = np.zeros(len(y))
    cv = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    for tr, te in cv.split(df, y, groups):
        m = make_model()
        m.fit(df.iloc[tr][cols], y[tr])
        p[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
    return p


def feature_importance(df, cols, y):
    """(feature, importance) sorted desc, from one RF fit on the whole dataset.

    Diagnostic only (not cross-validated) - used to pick which single features are
    worth an individual ablation. Column order is preserved through the median imputer,
    so importances line up with `cols`.
    """
    m = make_model()
    m.fit(df[cols], y)
    imp = m.named_steps["clf"].feature_importances_
    return sorted(zip(cols, imp), key=lambda kv: -kv[1])


def within_organ_auc(df, p, y, key="organ", min_n=WITHIN_MIN_N,
                     min_minority=WITHIN_MIN_MINORITY):
    """Mean AUC computed SEPARATELY inside each organ.

    Pooled AUC rewards ranking easy structures above hard ones. Within-organ AUC
    strips that out: it only asks whether, among masks of the SAME structure, the
    good ones score higher. A large pooled-minus-within gap means most of the
    apparent performance is organ difficulty, not mask quality.
    """
    out = []
    for name, g in df.assign(_p=p, _y=y).groupby(key):
        n_pos, n_neg = int(g._y.sum()), int((1 - g._y).sum())
        if len(g) >= min_n and min(n_pos, n_neg) >= min_minority:
            out.append({key: name, "n": len(g), "accept_rate": g._y.mean(),
                        "auc": roc_auc_score(g._y, g._p)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- experiments

def experiment_ablation(df, all_cols, y, groups):
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
        p = oof_predict(df, cols, y, groups)
        w = within_organ_auc(df, p, y)
        rows.append({"config": label, "kind": kind, "n_features": len(cols),
                     "auc": roc_auc_score(y, p),
                     "pr_auc_rej": average_precision_score(1 - y, 1 - p),
                     "within_organ_auc": w.auc.mean() if len(w) else np.nan,
                     "n_organs_within": len(w)})
        r = rows[-1]
        print(f"  {label:28s} feats={r['n_features']:>3d}  AUC={r['auc']:.3f}  "
              f"rej={r['pr_auc_rej']:.3f}  within-organ={r['within_organ_auc']:.3f}")

    print("\nfull model:")
    run("ALL", all_cols, "full")

    print("\nleave-one-group-out (how much is lost without this group):")
    for g, cs in groups_present.items():
        run(f"without {g}", [c for c in all_cols if c not in cs], "drop_group")

    print("\nonly-one-group (how much this group carries alone):")
    for g, cs in groups_present.items():
        run(f"only {g}", cs, "only_group")

    # ---- single-feature ablation for the top features by importance ----
    ranked_imp = feature_importance(df, all_cols, y)
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
    full = res[res.kind == "full"].iloc[0]
    res["delta_auc"] = res.auc - full.auc
    res["delta_within"] = res.within_organ_auc - full.within_organ_auc
    res.to_csv(os.path.join(OUT_DIR, "feature_ablation.csv"), index=False)

    # ---- unified table: every ablation, pooled + within-organ AUC and both deltas,
    #      sorted by pooled delta (most harmful first) ----
    print("\n  --- all ablations, sorted by pooled delta vs full model ---")
    print(f"  {'config':26s} {'pooledAUC':>9s} {'withinAUC':>9s} {'d_pooled':>9s} {'d_within':>9s}")
    print("  " + "-" * 66)
    ordered = res[res.kind != "full"].sort_values("delta_auc")
    for r in ordered.itertuples(index=False):
        print(f"  {r.config:26s} {r.auc:>9.3f} {r.within_organ_auc:>9.3f} "
              f"{r.delta_auc:>+9.3f} {r.delta_within:>+9.3f}")

    # ---- interpretation: which groups separate easy/hard structures rather than
    #      judging mask quality (hurt pooled much more than within-organ when dropped) ----
    print("\n  --- interpretation (dropped groups) ---")
    drops = res[res.kind == "drop_group"].sort_values("delta_auc")
    flagged = []
    for r in drops.itertuples(index=False):
        g = r.config.replace("without ", "")
        # more-negative delta_auc than delta_within => removing it costs pooled ranking
        # more than within-organ quality => the group was mostly an easy/hard separator.
        easy_hard_gap = r.delta_within - r.delta_auc
        tag = "  <- separates easy/hard, not good/bad" if easy_hard_gap > 0.02 else ""
        if easy_hard_gap > 0.02:
            flagged.append(g)
        print(f"  dropping {g:12s} pooled {r.delta_auc:+.3f}   within-organ {r.delta_within:+.3f}{tag}")
    print("\n  A group that hurts POOLED far more than WITHIN-ORGAN is mostly")
    print("  separating easy structures from hard ones, not good masks from bad.")
    if flagged:
        print(f"  flagged (pooled-dominant): {', '.join(flagged)}")
    else:
        print("  none flagged: every group that matters contributes to WITHIN-organ")
        print("  discrimination too, i.e. real mask-quality signal, not just easy/hard ranking.")
    return res


def experiment_threshold(df, all_cols, groups):
    print("\n" + "=" * 74)
    print("2. THRESHOLD SWEEP")
    print("=" * 74)
    rows = []
    print(f"\n{'IoU cut':>8s} {'accept%':>9s} {'AUC':>7s} {'rej PR':>8s} "
          f"{'within-organ':>13s} {'n organs':>9s}")
    for t in THRESHOLDS:
        y = (df["iou"] >= t).astype(int).to_numpy()
        if y.mean() in (0.0, 1.0):
            continue
        p = oof_predict(df, all_cols, y, groups)
        w = within_organ_auc(df, p, y)
        rows.append({"iou_threshold": t, "accept_rate": y.mean(),
                     "auc": roc_auc_score(y, p),
                     "pr_auc_rej": average_precision_score(1 - y, 1 - p),
                     "within_organ_auc": w.auc.mean() if len(w) else np.nan,
                     "n_organs_within": len(w)})
        r = rows[-1]
        print(f"{t:>8.2f} {100*r['accept_rate']:>8.1f}% {r['auc']:>7.3f} "
              f"{r['pr_auc_rej']:>8.3f} {r['within_organ_auc']:>13.3f} "
              f"{r['n_organs_within']:>9d}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "threshold_sweep.csv"), index=False)
    print("\n  Stable AUC across thresholds means 0.90 is not load-bearing.")
    print("  Note accept% shifts a lot, so PR-AUC moves with the class balance;")
    print("  compare AUC (no prevalence floor) across rows, not PR-AUC.")
    return res


def experiment_within(df, all_cols, y, groups):
    print("\n" + "=" * 74)
    print("3. WITHIN-ORGAN AND WITHIN-FAMILY AUC")
    print("=" * 74)
    p = oof_predict(df, all_cols, y, groups)
    pooled = roc_auc_score(y, p)

    w_org = within_organ_auc(df, p, y, key="organ")
    w_fam = within_organ_auc(df, p, y, key="family", min_n=FAMILY_MIN_N)

    print(f"\n  pooled AUC              {pooled:.3f}")
    print(f"  mean within-ORGAN AUC   {w_org.auc.mean():.3f}   ({len(w_org)} organs)")
    print(f"  mean within-FAMILY AUC  {w_fam.auc.mean():.3f}   ({len(w_fam)} families)")
    print(f"  pooled - within-organ   {pooled - w_org.auc.mean():+.3f}")
    print("\n  The gap is the share of pooled AUC that comes from ranking structures")
    print("  against each other rather than judging masks of the same structure.")

    w_org = w_org.assign(family=[family_of(o) for o in w_org.organ])
    w_org.sort_values("auc").to_csv(os.path.join(OUT_DIR, "within_organ_auc.csv"),
                                    index=False)
    w_fam.to_csv(os.path.join(OUT_DIR, "within_family_auc.csv"), index=False)

    print("\n  weakest organs (model barely discriminates within them):")
    print(w_org.nsmallest(6, "auc")[["organ", "n", "accept_rate", "auc"]]
          .to_string(index=False))
    print("\n  strongest:")
    print(w_org.nlargest(6, "auc")[["organ", "n", "accept_rate", "auc"]]
          .to_string(index=False))
    print("\n  by family:")
    print(w_fam.sort_values("auc").to_string(index=False))
    return w_org, w_fam


def leave_one_key_out(df, all_cols, key, min_n, min_minority):
    """Train on every level of `key` except one, test on the held-out level.

    Same model/feature set regardless of key, so organ-vs-family results are directly
    comparable and any difference is purely the granularity of what was held out.
    Returns a DataFrame with one row per held-out level.
    """
    groups_ = [(k, g) for k, g in df.groupby(key)
               if len(g) >= min_n and g["label"].nunique() == 2
               and min(g["label"].sum(), (1 - g["label"]).sum()) >= min_minority]
    rows = []
    for k, test in groups_:
        train = df[df[key] != k]
        m = make_model()
        m.fit(train[all_cols], train["label"])
        p = m.predict_proba(test[all_cols])[:, 1]
        rows.append({key: k, "n": len(test), "n_organs": test.organ.nunique(),
                     "accept_rate": test["label"].mean(),
                     "auc": roc_auc_score(test["label"], p)})
    return pd.DataFrame(rows)


def experiment_family_loo(df, all_cols, y, groups):
    print("\n" + "=" * 74)
    print("4. LEAVE-ONE-FAMILY-OUT")
    print("=" * 74)
    print("\n  Holding out ONE rib leaves 23 near-identical ribs in training, so")
    print("  per-organ LOO overstates transfer. This holds out the whole family.")

    res = leave_one_key_out(df, all_cols, "family", FAMILY_MIN_N, WITHIN_MIN_MINORITY)
    if res.empty:
        print("  no families with enough data")
        return None

    res = res.sort_values("auc")
    print(f"\n{'family':20s} {'n':>6s} {'organs':>7s} {'accept%':>8s} {'AUC':>7s}")
    print("-" * 52)
    for r in res.itertuples(index=False):
        print(f"{r.family:20s} {r.n:>6d} {r.n_organs:>7d} "
              f"{100*r.accept_rate:>7.1f}% {r.auc:>7.3f}")
    print("-" * 52)
    print(f"{'MEAN':20s} {'':>6s} {'':>7s} {'':>8s} {res.auc.mean():>7.3f}")
    print(f"{'MEDIAN':20s} {'':>6s} {'':>7s} {'':>8s} {res.auc.median():>7.3f}")
    print(f"\n  families above chance: {(res.auc > 0.5).sum()}/{len(res)}")
    res.to_csv(os.path.join(OUT_DIR, "family_loo.csv"), index=False)

    # ---- leave-one-ORGAN-out under the identical model/features, for a fair contrast ----
    org = leave_one_key_out(df, all_cols, "organ", WITHIN_MIN_N, WITHIN_MIN_MINORITY)
    org.to_csv(os.path.join(OUT_DIR, "organ_loo.csv"), index=False)
    print("\n  --- organ-LOO vs family-LOO (same model & features) ---")
    print(f"  leave-one-ORGAN-out    mean {org.auc.mean():.3f}   median {org.auc.median():.3f}   ({len(org)} organs)")
    print(f"  leave-one-FAMILY-out   mean {res.auc.mean():.3f}   median {res.auc.median():.3f}   ({len(res)} families)")
    print(f"  drop (organ - family)  mean {org.auc.mean() - res.auc.mean():+.3f}")
    print("\n  Organ-LOO is optimistic: holding out one rib leaves ~23 near-identical")
    print("  ribs in training (22 of the usable organs are ribs), so the model still")
    print("  sees the structure type. Family-LOO removes the whole type at once - the")
    print("  drop is the price of true novel-structure transfer, not a bug.")
    return res


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV)
    if "is_empty" in df:
        df = df[df["is_empty"] == 0].copy()
    df["label"] = (df["iou"] >= IOU_ACCEPT).astype(int)
    df["family"] = [family_of(o) for o in df["organ"]]
    df = df.reset_index(drop=True)

    if USE_REFERENCE and HAVE_REF and os.path.exists(REFERENCE_CSV):
        df = add_relative_features(df, load_reference(REFERENCE_CSV))

    all_cols = [c for c in df.columns if c not in LEAK_COLS + ID_COLS]
    # is_empty is constant once empty rows are dropped; drop any zero-variance column
    all_cols = [c for c in all_cols if df[c].nunique(dropna=False) > 1]
    # never let a ground-truth-derived column reach the feature matrix
    assert not (set(all_cols) & set(LEAK_COLS)), f"leak: {set(all_cols) & set(LEAK_COLS)}"

    y = df["label"].to_numpy()
    groups = df["subject"].to_numpy()

    print(f"{len(df)} masks | {df.subject.nunique()} subjects | "
          f"{df.organ.nunique()} organs | {df.family.nunique()} families | "
          f"{100*y.mean():.1f}% accept")
    print(f"{len(all_cols)} features\n")
    fam_counts = df.family.value_counts()
    print("family sizes:")
    for f, n in fam_counts.items():
        print(f"  {f:20s} {n:>5d}  ({df[df.family==f].organ.nunique()} organs)")
    if "other" in fam_counts:
        unmapped = sorted(df[df.family == "other"].organ.unique())[:15]
        print(f"\n  unmapped -> 'other': {unmapped}")
        print("  (extend FAMILY_PATTERNS if any of these should be grouped)")

    if RUN_ABLATION:
        experiment_ablation(df, all_cols, y, groups)
    if RUN_THRESHOLD:
        experiment_threshold(df, all_cols, groups)
    if RUN_WITHIN:
        experiment_within(df, all_cols, y, groups)
    if RUN_FAMILY_LOO:
        experiment_family_loo(df, all_cols, y, groups)

    print(f"\nall CSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()