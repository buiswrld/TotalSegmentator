"""
Derive the small paper-final summaries that were missing as CSVs (audit gaps G7, G1, G6)
plus the like-for-like silent-failure rates. READ-ONLY on experiments: no model is
retrained for G7 (bootstraps saved OOF predictions); G1 subsets an existing table; G6
fits a classifier only on the CT val / MRI dev subset (never test) to produce calibration
curves. All outputs land in paper_final/.

  G7  primary_model_summary.csv (ct + mri): OOF AUC with 95% SUBJECT-bootstrap CI
      (resample subjects with replacement, grouped) + PR-AUC per class + minority mark.
  G1  mri/dataset_combined_metrics.csv + mri/dataset_summary_by_organ.csv: the MRI dev
      subset (carved exactly as regression_modality.py did) and its per-structure stats.
  G6  mri/calibration_metrics.csv (+ _no_intensity): ECE/MCE/Brier decomposition,
      uncalibrated vs isotonic, nested subject-grouped, both intensity variants.
  silent_failure_summary.csv: raw rate (vol_err<0.10 & IoU<0.90) AND conditional rate
      (of IoU<0.90 masks, fraction with vol_err<0.10), CT and MRI, like-for-like.

Run:  python experiments/pipeline/paper_consolidate.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression as R
from common import ece_mce, brier_decomposition

A = r"C:\Users\ansar\Algoverse"
FINAL = os.path.join(A, "paper_final"); CT = os.path.join(FINAL, "ct"); MRI = os.path.join(FINAL, "mri")
MRI_TRAIN_CSV = os.path.abspath("experiments/eval_runs/mr_full_run/metrics/train/combined_metrics.csv")
INTENSITY_COLS = ["mean_HU", "median_HU", "std_HU", "p05_HU", "p95_HU"]
N_BOOT = 1000
SEED = 0


# ---------------------------------------------------------------- G7: bootstrap CIs

def bootstrap_auc_ci(y, score, subjects, n_boot=N_BOOT, seed=SEED):
    """95% CI for ROC-AUC by resampling SUBJECTS with replacement (grouped bootstrap)."""
    rng = np.random.default_rng(seed)
    subs = np.unique(subjects)
    idx_by_sub = {s: np.where(subjects == s)[0] for s in subs}
    aucs = []
    for _ in range(n_boot):
        samp = rng.choice(subs, size=len(subs), replace=True)
        rows = np.concatenate([idx_by_sub[s] for s in samp])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, score[rows]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def organ_rate_oof(df):
    """Subject-grouped OOF organ-rate baseline: predict each organ's accept rate learned
    from the training folds only, with NO mask geometry - the 'beats organ identity'
    control. Mirrors train.py::organ_baseline. Returns per-row predicted accept rate."""
    y = (df["iou"].to_numpy() >= R.IOU_ACCEPT).astype(int)
    groups = df["subject"].to_numpy()
    p = np.zeros(len(df))
    lab = pd.Series(y, index=df.index)
    cv = GroupKFold(n_splits=min(R.N_SPLITS, len(np.unique(groups))))
    for tr, te in cv.split(df, y, groups):
        rates = lab.iloc[tr].groupby(df.iloc[tr]["organ"].values).mean()
        p[te] = df.iloc[te]["organ"].map(rates).fillna(y[tr].mean()).to_numpy()
    return p


def primary_summary(df, model_cols, out_csv, note):
    """One row per model/score: point AUC, 95% subject-bootstrap CI, PR-AUC both classes."""
    y = (df["iou"].to_numpy() >= R.IOU_ACCEPT).astype(int) if "label" not in df else df["label"].to_numpy()
    subjects = df["subject"].to_numpy()
    minority = "accept" if y.mean() < 0.5 else "reject"
    rows = []
    for col in model_cols:
        s = df[col].to_numpy(dtype=float)
        if len(np.unique(s)) < 2:                      # e.g. always-accept constant
            auc, lo, hi = 0.5, np.nan, np.nan
        else:
            auc = roc_auc_score(y, s)
            lo, hi = bootstrap_auc_ci(y, s, subjects)
        pr_acc = average_precision_score(y, s)
        pr_rej = average_precision_score(1 - y, -s)
        rows.append({"model": col, "n": len(df), "accept_rate": round(float(y.mean()), 4),
                     "auc": round(auc, 4), "auc_ci_lo": round(lo, 4) if lo == lo else "",
                     "auc_ci_hi": round(hi, 4) if hi == hi else "",
                     "pr_auc_accept": round(pr_acc, 4), "pr_auc_reject": round(pr_rej, 4),
                     "minority_class": minority,
                     "pr_auc_minority": round(pr_acc if minority == "accept" else pr_rej, 4)})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out.to_csv(out_csv, index=False)
    print(f"  [{note}] wrote {os.path.basename(out_csv)} ({len(out)} models); "
          f"best {out.iloc[0]['model']} AUC {out.iloc[0]['auc']} "
          f"[{out.iloc[0]['auc_ci_lo']}, {out.iloc[0]['auc_ci_hi']}]")
    return out


# ---------------------------------------------------------------- G1: MRI dev stats

def carve_mri_dev():
    """Reproduce regression_modality.py's dev carve EXACTLY: normalize, then a single
    subject-grouped 20% split at seed 0. Returns the dev-subset dataframe (short schema)."""
    mri = R.normalize_columns(pd.read_csv(MRI_TRAIN_CSV))
    subs = mri["subject"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
    _, dev_idx = next(gss.split(mri, mri["iou"], subs))
    return mri.iloc[dev_idx].copy().reset_index(drop=True)


def summary_by_organ(df):
    """Per-structure rollup matching compute_metrics.py's summary_by_organ semantics:
    n_scored/mean_dice/mean_iou over rows present in both (status ok|low_dice); accept_rate
    over ALL rows of that organ (incl miss/false_positive)."""
    rows = []
    for organ, g in df.groupby("organ"):
        scored = g[g["status"].isin(["ok", "low_dice"])]
        n_acc = int((g["iou"] >= R.IOU_ACCEPT).sum())
        rows.append({"organ": organ, "n_scored": len(scored),
                     "mean_dice": round(scored["dice"].mean(), 4) if len(scored) else "",
                     "mean_iou": round(scored["iou"].mean(), 4) if len(scored) else "",
                     "n_accept": n_acc, "accept_rate": round(n_acc / len(g), 4)})
    return pd.DataFrame(rows).sort_values("organ")


# ---------------------------------------------------------------- G6: MRI calibration

def rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=R.N_TREES, min_samples_leaf=2,
                             class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])


def oof_probs(df, cols, y, groups, calib):
    """Subject-grouped OOF probabilities. calib in {'none','isotonic','sigmoid'}; the
    isotonic/sigmoid (Platt) map is fit on a subject-disjoint inner split of each
    outer-training fold (nested, no leakage). Platt = 1-parameter logistic on the score,
    expected to survive a small-sample regime where isotonic (non-parametric) overfits."""
    p = np.zeros(len(df))
    outer = GroupKFold(n_splits=min(R.N_SPLITS, len(np.unique(groups))))
    for tr, te in outer.split(df, y, groups):
        if calib == "none":
            m = rf(); m.fit(df.iloc[tr][cols], y[tr])
            p[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
            continue
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
        fit_rel, cal_rel = next(gss.split(df.iloc[tr], y[tr], groups[tr]))
        fit_idx, cal_idx = tr[fit_rel], tr[cal_rel]
        m = rf(); m.fit(df.iloc[fit_idx][cols], y[fit_idx])
        cal_p = m.predict_proba(df.iloc[cal_idx][cols])[:, 1]
        te_p = m.predict_proba(df.iloc[te][cols])[:, 1]
        if calib == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(cal_p, y[cal_idx])
            p[te] = iso.predict(te_p)
        else:  # sigmoid / Platt
            lr = LogisticRegression(); lr.fit(cal_p.reshape(-1, 1), y[cal_idx])
            p[te] = lr.predict_proba(te_p.reshape(-1, 1))[:, 1]
    return p


def calibration_metrics(df, cols, out_csv, note):
    y = (df["iou"].to_numpy() >= R.IOU_ACCEPT).astype(int)
    groups = df["subject"].to_numpy()
    rows = []
    for calib in ("none", "isotonic", "sigmoid"):
        p = oof_probs(df, cols, y, groups, calib=calib)
        ece, mce, _ = ece_mce(y.astype(float), p)
        bd = brier_decomposition(y.astype(float), p)
        rows.append({"model": "RF (class_weight=balanced)", "calibration": calib,
                     "auc": round(roc_auc_score(y, p), 4), "ece": round(ece, 4),
                     "mce": round(mce, 4), "brier": round(bd["brier"], 4),
                     "reliability": round(bd["reliability"], 6),
                     "resolution": round(bd["resolution"], 6),
                     "uncertainty": round(bd["uncertainty"], 6)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [{note}] wrote {os.path.basename(out_csv)}: "
          + "  ".join(f"{r['calibration']} ECE {r['ece']}" for r in rows))


# ---------------------------------------------------------------- silent failure

def silent_failure_row(modality, oof_csv, in_sample):
    d = pd.read_csv(oof_csv)
    total = len(d)
    low_iou = int((d["iou"] < R.IOU_ACCEPT).sum())
    silent = int(((d["vol_err"] < 0.10) & (d["iou"] < R.IOU_ACCEPT)).sum())
    return {"modality": modality, "in_sample": in_sample, "n_masks_gt_present": total,
            "n_low_iou": low_iou, "n_silent_failure": silent,
            "raw_rate": round(silent / total, 4),
            "conditional_rate_of_low_iou": round(silent / low_iou, 4)}


# ---------------------------------------------------------------- main

def main():
    print("G7: primary-model AUC + 95% subject-bootstrap CI (no retraining)")
    ct_oof = pd.read_csv(os.path.join(CT, "primary_model_oof.csv"))
    ct_models = [c for c in ct_oof.columns if c not in ("subject", "organ", "iou", "label")]
    primary_summary(ct_oof, ct_models, os.path.join(CT, "primary_model_summary.csv"), "CT")

    mri_oof = pd.read_csv(os.path.join(MRI, "iou_regression_oof.csv"))
    mri_oof["organ-rate_baseline"] = organ_rate_oof(mri_oof)   # G2: the 'beats organ identity' anchor
    primary_summary(mri_oof, ["pred_iou_RandomForest", "pred_iou_HistGB", "organ-rate_baseline"],
                    os.path.join(MRI, "primary_model_summary.csv"), "MRI-dev")

    print("\nG1: MRI dev dataset stats (subset carve, no inference)")
    dev = carve_mri_dev()
    dev.to_csv(os.path.join(MRI, "dataset_combined_metrics.csv"), index=False)
    sbo = summary_by_organ(dev)
    sbo.to_csv(os.path.join(MRI, "dataset_summary_by_organ.csv"), index=False)
    ne = dev[dev.is_empty == 0]
    print(f"  MRI dev: {len(dev)} masks, {int((dev.is_empty==1).sum())} empty, {len(ne)} modelled, "
          f"{dev.subject.nunique()} subjects, {dev.organ.nunique()} structures; "
          f"accept@0.90 all={ (dev.iou>=0.90).mean():.4f} nonempty={(ne.iou>=0.90).mean():.4f}")

    print("\nG6: MRI dev calibration (nested grouped, none vs isotonic)")
    cols_all = R.feature_columns(ne)
    calibration_metrics(ne, cols_all, os.path.join(MRI, "calibration_metrics.csv"), "MRI +intensity")
    cols_noint = [c for c in cols_all if c not in INTENSITY_COLS]
    calibration_metrics(ne, cols_noint, os.path.join(MRI, "calibration_metrics_no_intensity.csv"), "MRI -intensity")

    print("\nSilent-failure rates (raw + conditional), like-for-like")
    rows = [silent_failure_row("CT (val, held out)", os.path.join(CT, "volume_label_oof.csv"), False),
            silent_failure_row("MRI (dev, IN-SAMPLE)", os.path.join(MRI, "volume_label_oof.csv"), True)]
    sf = pd.DataFrame(rows)
    sf.to_csv(os.path.join(FINAL, "silent_failure_summary.csv"), index=False)
    print(sf.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
