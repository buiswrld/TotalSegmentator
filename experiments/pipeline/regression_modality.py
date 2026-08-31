"""
Modality-aware driver for the regression suite (regression.py), run separately for CT
and MRI and reported side by side. This is an ADDITIVE orchestrator: it imports and calls
regression.py's task functions unchanged (so CT reproduces the standalone run) and layers
on the CT/MRI differences the standalone script doesn't handle.

What it adds over regression.py:
  - reads BOTH schemas via regression.normalize_columns (long descriptive MRI labels ->
    short CT names); the leak-guard assert still runs on the normalized names.
  - MRI eval set: now TotalSegmentator's own official `test` split (49/55 subjects
    covered - see experiments/pipeline/split_by_official.py), genuinely held out and
    never touched by training/CV, exactly like CT's val/test. This replaces an earlier
    version of this script that carved an in-sample 20% dev subset out of MRI *train*
    (kept only as a historical note: that approach was used because no full-dataset
    MRI test predictions existed yet at the time).
  - class balance is per modality: MRI accept is the MINORITY (~17-27%) vs CT's 74%. The
    direct-classifier baseline AUC is recomputed FRESH per modality (the CT 0.870 constant
    is never used for MRI); we report BOTH classes' PR-AUC and mark the minority; and for
    MRI we report the direct classifier with and without balanced class weighting.
  - MRI intensity (mean/median/std/p05/p95 "HU") is uncalibrated and per-scan arbitrary, so
    every MRI experiment is run twice: with and without the intensity feature group.
  - FAMILY_PATTERNS rebuilt for the 50-structure total_mr set (intervertebral_discs folded
    into the spine/vertebrae family; everything else already maps).
  - Mondrian conformal uses the feasibility floor (n>=4 for 80%, n>=9 for 90%) via
    regression.MONDRIAN_FLOOR_MODE="feasibility".

Run:  python experiments/pipeline/regression_modality.py
"""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression as R
from ablations import FAMILY_PATTERNS

# ---------------------------------------------------------------- paths / config
# Repointed at the full-dataset, official-TotalSegmentator-split run (see
# experiments/pipeline/split_by_official.py and RESEARCH.md) - both CT and MRI now use
# their real held-out test set directly, no more MRI dev-carving.
CT_CSV   = "experiments/eval_runs/ct_official_split/metrics/official_test_combined_metrics.csv"
MRI_CSV  = "experiments/eval_runs/mri_official_split/metrics/official_test_combined_metrics.csv"  # real held-out
OUT_ROOT = "experiments/paper_final_v2"
CT_OUT   = os.path.join(OUT_ROOT, "ct", "regression")
MRI_OUT_INT   = os.path.join(OUT_ROOT, "mri", "regression_intensity")
MRI_OUT_NOINT = os.path.join(OUT_ROOT, "mri", "regression_no_intensity")

SEED = 0
# intensity descriptors (short CT names post-normalization); uncalibrated on MRI
INTENSITY_COLS = ["mean_HU", "median_HU", "std_HU", "p05_HU", "p95_HU"]

# MRI family mapping: reuse the CT patterns, add the one MRI structure that otherwise
# lands in "other". Intervertebral discs are spinal soft tissue -> group with vertebrae.
MRI_FAMILY_PATTERNS = FAMILY_PATTERNS + [("vertebrae", r"^intervertebral_disc")]


def family_of_mri(organ):
    for name, pat in MRI_FAMILY_PATTERNS:
        if re.search(pat, organ):
            return name
    return "other"


# ---------------------------------------------------------------- baselines / metrics

def direct_classifier_auc(df, cols, groups, balanced=True):
    """Fresh direct accept/reject classifier baseline: grouped-OOF RandomForest on the
    label (iou>=0.90), returning ROC-AUC. This replaces the hardcoded CT 0.870 constant
    and is recomputed per modality / per feature set."""
    y = (df["iou"].to_numpy() >= R.IOU_ACCEPT).astype(int)
    p = np.zeros(len(df))
    cv = GroupKFold(n_splits=min(R.N_SPLITS, len(np.unique(groups))))
    for tr, te in cv.split(df, y, groups):
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean(); continue
        clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("clf", RandomForestClassifier(
                            n_estimators=R.N_TREES, min_samples_leaf=2,
                            class_weight=("balanced_subsample" if balanced else None),
                            random_state=SEED, n_jobs=-1))])
        clf.fit(df.iloc[tr][cols], y[tr])
        p[te] = clf.predict_proba(df.iloc[te][cols])[:, 1]
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def both_class_pr(iou, score, t=R.IOU_ACCEPT):
    """PR-AUC for the accept and reject classes, and which one is the minority."""
    y = (iou >= t).astype(int)
    pr_accept = average_precision_score(y, score)
    pr_reject = average_precision_score(1 - y, -score)
    minority = "accept" if y.mean() < 0.5 else "reject"
    return {"accept_rate": float(y.mean()), "pr_auc_accept": pr_accept,
            "pr_auc_reject": pr_reject, "minority_class": minority,
            "pr_auc_minority": pr_accept if minority == "accept" else pr_reject}


# ---------------------------------------------------------------- one modality run

def run_modality(tag, df_all, out_dir, family_fn, drop_intensity):
    """Set regression.py's module globals for this modality/variant and run all four
    tasks against the given (already normalized, dev-subset) dataframe. Returns a compact
    result dict for the side-by-side summary."""
    os.makedirs(out_dir, exist_ok=True)
    R.OUT_DIR = out_dir
    R.family_of = family_fn                 # tasks look up module-global family_of at call time
    R.MONDRIAN_FLOOR_MODE = "feasibility"

    df_all = df_all.copy()
    df_all["family"] = [family_fn(o) for o in df_all["organ"]]
    n_empty = int((df_all.is_empty == 1).sum())
    df = df_all[df_all.is_empty == 0].copy().reset_index(drop=True)
    groups = df["subject"].to_numpy()

    cols = R.feature_columns(df)
    if drop_intensity:
        cols = [c for c in cols if c not in INTENSITY_COLS]
    assert not (set(cols) & set(R.LEAK_COLS)), "leak guard"

    y = (df["iou"].to_numpy() >= R.IOU_ACCEPT).astype(int)
    print("\n" + "#" * 78)
    print(f"# {tag}")
    print("#" * 78)
    print(f"{len(df_all)} masks | {n_empty} empty dropped (trivial rule) | {len(df)} modelled | "
          f"{df.subject.nunique()} subjects | {df.organ.nunique()} structures | "
          f"{df.family.nunique()} families | {len(cols)} features")
    print(f"accept rate (iou>=0.90): {100*y.mean():.1f}%  -> minority class: "
          f"{'accept' if y.mean()<0.5 else 'reject'}")

    # direct classifier baseline (fresh), with and without balanced weighting
    auc_bal = direct_classifier_auc(df, cols, groups, balanced=True)
    auc_unbal = direct_classifier_auc(df, cols, groups, balanced=False)
    print(f"direct classifier AUC  balanced={auc_bal:.3f}  unbalanced={auc_unbal:.3f}  "
          f"(delta {auc_bal-auc_unbal:+.3f})")
    R.DIRECT_CLASSIFIER_AUC = auc_bal        # task1 compares the regressor against this

    # --- Task 1 ---
    pred_iou = R.task1_iou_regression(df, cols, groups)
    pr = both_class_pr(df["iou"].to_numpy(), pred_iou)
    print(f"\n  [both-class PR-AUC @0.90]  accept {pr['pr_auc_accept']:.3f}  "
          f"reject {pr['pr_auc_reject']:.3f}  (minority = {pr['minority_class']}, "
          f"PR-AUC={pr['pr_auc_minority']:.3f})")

    # --- Task 2 (+2b conformal) ---
    R.task2_quantile_intervals(df, cols, groups)
    R.task2b_conformal(df, cols, groups)

    # --- Task 3 ---
    dv = df[df.gt_vox > 0].copy().reset_index(drop=True)
    print(f"\n(volume tasks: dropped {len(df)-len(dv)} gt_vox==0 false-positive rows)")
    R.task3_volume_label(df_all, cols, groups, dv, dv["subject"].to_numpy())

    # --- Task 4 ---
    R.task4_clinical_flips(df, pred_iou)

    return {"tag": tag, "out_dir": out_dir, "n": len(df), "subjects": df.subject.nunique(),
            "accept_rate": float(y.mean()), "direct_auc_bal": auc_bal,
            "direct_auc_unbal": auc_unbal, "regress_pr": pr, "cols": len(cols)}


# ---------------------------------------------------------------- size-bias tolerance

def size_tolerance_table(df, out_csv, iou_thr=R.IOU_ACCEPT):
    """Per-organ boundary tolerance implied by the IoU threshold: for a sphere of the
    organ's median volume, IoU~=1-3d/r => d = r*(1-thr)/3. Reported in mm AND in voxels
    (using each organ's median voxel size) - recomputed from THIS modality's spacing."""
    d = df[df.gt_vox > 0].copy()
    d["voxvol"] = d.volume_mm3 / d.num_voxels
    d["gt_mm3"] = d.gt_vox * d.voxvol
    rows = []
    for o, g in d.groupby("organ"):
        if len(g) < 3:
            continue
        V = g.gt_mm3.median()
        r = (3 * V / (4 * np.pi)) ** (1 / 3)
        dd = r * (1 - iou_thr) / 3
        sp = g.voxvol.median() ** (1 / 3)
        rows.append({"organ": o, "n": len(g), "median_vol_mL": round(V / 1000, 2),
                     "radius_mm": round(r, 2), "tol_mm": round(dd, 3),
                     "spacing_mm": round(sp, 3), "tol_voxels": round(dd / sp, 3)})
    t = pd.DataFrame(rows).sort_values("median_vol_mL", ascending=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    t.to_csv(out_csv, index=False)
    return t


# ---------------------------------------------------------------- main

def main():
    results = []

    # ---------------- CT (val, held out) ----------------
    ct_all = R.normalize_columns(pd.read_csv(CT_CSV))
    print("\n" + "=" * 78)
    print("MRI family table is reported under the MRI section; CT uses the 117-structure map.")
    st_ct = size_tolerance_table(ct_all[ct_all.is_empty == 0], os.path.join(CT_OUT, "size_tolerance.csv"))
    results.append(run_modality("CT  (official val+test, genuinely held out)", ct_all, CT_OUT,
                                R.family_of, drop_intensity=False))

    # ---------------- MRI (official test split, genuinely held out) ----------------
    mri_dev = R.normalize_columns(pd.read_csv(MRI_CSV))

    # MRI family table (state what maps where and what would fall in "other")
    fam = pd.Series([family_of_mri(o) for o in sorted(mri_dev["organ"].unique())],
                    index=sorted(mri_dev["organ"].unique()))
    print("\n" + "=" * 78)
    print("MRI FAMILY TABLE (total_mr, 50 structures; intervertebral_discs folded into vertebrae/spine)")
    print("=" * 78)
    for f in sorted(fam.unique()):
        members = sorted(fam[fam == f].index)
        print(f"  {f:16s} ({len(members)}): {', '.join(members)}")
    orphan = [o for o in fam.index if family_of_mri(o) == "other"]
    print(f"  -> 'other': {orphan if orphan else 'none'}")

    st_mri = size_tolerance_table(mri_dev[mri_dev.is_empty == 0],
                                  os.path.join(MRI_OUT_INT, "size_tolerance.csv"))

    results.append(run_modality("MRI (official test, WITH intensity) - genuinely held out",
                                mri_dev, MRI_OUT_INT, family_of_mri, drop_intensity=False))
    results.append(run_modality("MRI (official test, NO intensity) - intensity uncalibrated on MRI",
                                mri_dev, MRI_OUT_NOINT, family_of_mri, drop_intensity=True))

    # ---------------- side-by-side ----------------
    print("\n" + "=" * 78)
    print("SIZE-BIAS TOLERANCE at IoU 0.90 (mm and voxels), key abdominal organs")
    print("=" * 78)
    key = ["liver", "spleen", "kidney_left", "pancreas", "adrenal_gland_left", "gallbladder"]
    print(f"  {'organ':20s} {'CT tol_mm':>9s} {'CT vox':>7s} {'MRI tol_mm':>10s} {'MRI vox':>8s}")
    for o in key:
        c = st_ct[st_ct.organ == o]; m = st_mri[st_mri.organ == o]
        cm = f"{c.tol_mm.iloc[0]:.3f}" if len(c) else "-"
        cv = f"{c.tol_voxels.iloc[0]:.3f}" if len(c) else "-"
        mm = f"{m.tol_mm.iloc[0]:.3f}" if len(m) else "-"
        mv = f"{m.tol_voxels.iloc[0]:.3f}" if len(m) else "-"
        print(f"  {o:20s} {cm:>9s} {cv:>7s} {mm:>10s} {mv:>8s}")

    print("\n" + "=" * 78)
    print("SIDE BY SIDE: direct classifier + IoU-regression (Task 1)")
    print("=" * 78)
    print(f"  {'run':40s} {'accept%':>8s} {'directAUC':>10s} {'PRacc':>7s} {'PRrej':>7s} {'minority':>9s}")
    for r in results:
        pr = r["regress_pr"]
        print(f"  {r['tag'][:40]:40s} {100*r['accept_rate']:>7.1f}% {r['direct_auc_bal']:>10.3f} "
              f"{pr['pr_auc_accept']:>7.3f} {pr['pr_auc_reject']:>7.3f} {pr['minority_class']:>9s}")

    # conformal coverage side-by-side (read each run's conformal_summary.csv)
    print("\n" + "=" * 78)
    print("SIDE BY SIDE: conformal marginal coverage (Task 2) - CQR global clipped")
    print("=" * 78)
    print(f"  {'run':40s} {'80% cov':>8s} {'80% medW':>9s} {'90% cov':>8s} {'90% medW':>9s}")
    for r in results:
        f = os.path.join(r["out_dir"], "conformal_summary.csv")
        if not os.path.exists(f):
            continue
        s = pd.read_csv(f)
        s = s[s.variant == "CQR global (clipped [0,1])"]
        def g(a, c):
            row = s[np.isclose(s.alpha, a)]
            return row[c].iloc[0] if len(row) else float("nan")
        print(f"  {r['tag'][:40]:40s} {100*g(0.2,'coverage'):>7.1f}% {g(0.2,'median_width'):>9.3f} "
              f"{100*g(0.1,'coverage'):>7.1f}% {g(0.1,'median_width'):>9.3f}")

    pd.DataFrame([{"run": r["tag"], "n": r["n"], "subjects": r["subjects"],
                   "accept_rate": r["accept_rate"], "direct_auc_balanced": r["direct_auc_bal"],
                   "direct_auc_unbalanced": r["direct_auc_unbal"],
                   "pr_auc_accept": r["regress_pr"]["pr_auc_accept"],
                   "pr_auc_reject": r["regress_pr"]["pr_auc_reject"],
                   "minority_class": r["regress_pr"]["minority_class"]} for r in results]
                 ).to_csv(os.path.join(OUT_ROOT, "modality_comparison.csv"), index=False)
    print(f"\nwrote modality_comparison.csv + per-run CSVs under {CT_OUT} and report_mri_dev/")


if __name__ == "__main__":
    main()
