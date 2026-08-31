"""
Regression / clinical-cost reframing of the mask quality-control problem.

The headline classifier predicts accept = (IoU >= 0.90). Two weaknesses this script
attacks, without touching the classifier:

  (a) 0.90 is an arbitrary cutoff, and binarising throws away most of the label signal
      (IoU 0.899 and IoU 0.30 are both just "reject"). -> TASK 1/2: regress IoU directly,
      threshold at inference for ANY cutoff, and put an interval around each prediction.
  (b) IoU is not a clinical quantity - nothing says what a bad mask COSTS. -> TASK 3/4:
      a volume-error label (clinically meaningful, computable from the same CSV) and a
      decision-flip analysis against published volumetric clinical thresholds.

Four tasks, each guarded by a RUN_* flag:

  1. IOU REGRESSION        predict IoU; does regress-then-threshold match direct
                           classification? one model serving every threshold.
  2. QUANTILE INTERVALS    q10/q50/q90 quantile regression; empirical coverage of the
                           80% interval, per-family width, width-vs-error correlation.
  3. VOLUME-ERROR LABEL    vol_bad = |pred_vox - gt_vox|/gt_vox > tau; does size become
                           load-bearing when the label is a volume function? what does a
                           volume-only check miss (sized-right-but-displaced masks)?
  4. CLINICAL DECISION     how often predicted volume flips a published clinical
     FLIPS                 threshold (splenomegaly/...), and how much filtering removes.

Subject-grouped CV throughout (GroupKFold on subject); never split by row. Ground-truth
columns (dice/iou/tp/fp/fn/gt_vox/status/accept) never enter the feature matrix - an
assert enforces it - even though gt_vox is used to BUILD the task-3/4 labels.

Run:  python experiments/pipeline/regression.py
"""

import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import os
import sys

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.metrics import (roc_auc_score, average_precision_score, mean_absolute_error,
                            mean_squared_error, r2_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablations import family_of, FEATURE_GROUPS
from common import ece_mce

# ============================ CONFIG ============================
CSV        = r"C:\Users\ansar\Algoverse\report_ct_val\combined_metrics.csv"
OUT_DIR    = r"C:\Users\ansar\Algoverse\report_ct_val\regression"

IOU_ACCEPT = 0.90
N_SPLITS   = 5
N_TREES    = 300
SEED       = 0

DIRECT_CLASSIFIER_AUC = 0.870   # headline accept=(IoU>=0.90) classifier, for comparison
THRESHOLDS = [0.70, 0.80, 0.85, 0.90, 0.95]
VOL_TAUS   = [0.10, 0.20]

RUN_IOU_REG    = True
RUN_QUANTILE   = True
RUN_VOLUME     = True
RUN_CLINICAL   = True
RUN_CONFORMAL  = True

# conformal calibration (Task 2b)
CONF_ALPHAS      = [0.2, 0.1]   # nominal miscoverage: 80% and 90% intervals
CONF_CAL_FRAC    = 0.30         # fraction of each outer-train's SUBJECTS held for calibration
MONDRIAN_MIN_CAL = 50           # min per-family calibration masks (used when MONDRIAN_FLOOR_MODE=="fixed")
# "fixed" -> require >= MONDRIAN_MIN_CAL calibration masks per family; "feasibility" ->
# require only the minimum n at which the conformal quantile is defined for this alpha
# (ceil((1-alpha)/alpha): n>=4 for 80%, n>=9 for 90%), families below it fall back to
# the global Q deterministically. The modality driver (regression_modality.py) sets
# "feasibility"; standalone CT runs keep "fixed" so earlier outputs are unchanged.
MONDRIAN_FLOOR_MODE = "fixed"
# ================================================================

LEAK_COLS = ["dice", "iou", "tp", "fp", "fn", "gt_vox", "status", "accept"]
ID_COLS   = ["subject", "organ", "orig_axcodes", "pred_vox", "family",
             "label", "vol_err", "vol_bad", "iou_reject"]   # never features


def normalize_columns(df):
    """Modality-aware schema shim: map the staged pipeline's long descriptive labels
    (e.g. 'Intersection over Union (IoU)') to the short CT names this module reads
    ('iou'), leaving already-short CT columns untouched. Only the ground-truth columns
    differ in case between the two schemas (IoU/Dice/TP/FP/FN vs iou/dice/tp/fp/fn);
    everything else (mean_HU, centroid_x_rel, ...) is identical once the parenthetical is
    extracted, so we preserve case except for those five. Idempotent on CT input.
    """
    import re
    def norm(c):
        m = re.search(r"\(([^)]+)\)\s*$", c)
        s = m.group(1) if m else c
        return s.lower() if s.lower() in {"iou", "dice", "tp", "fp", "fn"} else s
    return df.rename(columns={c: norm(c) for c in df.columns})


def mondrian_floor(alpha):
    """Minimum per-family calibration count to trust a family-specific conformal Q.

    'feasibility' mode uses the smallest n at which conformal_Q is even defined for this
    alpha: ceil((n+1)(1-alpha)) <= n  <=>  n >= ceil((1-alpha)/alpha)  (4 for 80%, 9 for
    90%). 'fixed' mode uses MONDRIAN_MIN_CAL regardless of alpha.
    """
    if MONDRIAN_FLOOR_MODE == "feasibility":
        return int(np.ceil((1 - alpha) / alpha))
    return MONDRIAN_MIN_CAL


def feature_columns(df):
    """Ground-truth-free feature columns, with the leak guard from train_classifier.py."""
    cols = [c for c in df.columns if c not in LEAK_COLS + ID_COLS]
    cols = [c for c in cols if df[c].nunique(dropna=False) > 1]   # drop zero-variance
    assert not (set(cols) & set(LEAK_COLS)), f"leak: {set(cols) & set(LEAK_COLS)}"
    return cols


def make_reg(kind):
    if kind == "histgb":
        clf = HistGradientBoostingRegressor(max_iter=400, random_state=SEED)
    else:
        clf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2,
                                    random_state=SEED, n_jobs=-1)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", clf)])


def make_clf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                             class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))])


def oof_regress(df, cols, target, groups, kind):
    """Subject-grouped out-of-fold continuous predictions."""
    p = np.zeros(len(df))
    cv = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    y = df[target].to_numpy()
    for tr, te in cv.split(df, y, groups):
        m = make_reg(kind)
        m.fit(df.iloc[tr][cols], y[tr])
        p[te] = m.predict(df.iloc[te][cols])
    return p


def oof_quantile(df, cols, groups, q):
    """Subject-grouped OOF predictions from a quantile-loss gradient booster."""
    p = np.zeros(len(df))
    cv = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    y = df["iou"].to_numpy()
    for tr, te in cv.split(df, y, groups):
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("reg", HistGradientBoostingRegressor(loss="quantile", quantile=q,
                              max_iter=400, random_state=SEED))])
        m.fit(df.iloc[tr][cols], y[tr])
        p[te] = m.predict(df.iloc[te][cols])
    return p


def oof_proba(df, cols, target, groups):
    """Subject-grouped OOF class-1 probabilities."""
    p = np.zeros(len(df))
    cv = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    y = df[target].to_numpy()
    for tr, te in cv.split(df, y, groups):
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        m = make_clf()
        m.fit(df.iloc[tr][cols], y[tr])
        p[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
    return p


def binary_from_score(iou, score, t):
    """ROC-AUC and PR-AUC(reject) for label (iou>=t), ranking masks by `score` (predicted
    IoU). Higher score = more likely accept, so reject ranks on -score."""
    y = (iou >= t).astype(int)
    if y.min() == y.max():
        return np.nan, np.nan, y.mean()
    return roc_auc_score(y, score), average_precision_score(1 - y, -score), y.mean()


# ---------------------------------------------------------------- Task 1

def task1_iou_regression(df, cols, groups):
    print("\n" + "=" * 74)
    print("1. IoU REGRESSION")
    print("=" * 74)
    iou = df["iou"].to_numpy()

    preds = {}
    print(f"\n{'model':14s} {'Spearman':>9s} {'Pearson':>8s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s}")
    print("-" * 58)
    for kind, label in [("histgb", "HistGB"), ("rf", "RandomForest")]:
        p = oof_regress(df, cols, "iou", groups, kind)
        p = np.clip(p, 0, 1)
        preds[label] = p
        sp = spearmanr(iou, p).correlation
        pe = pearsonr(iou, p)[0]
        mae = mean_absolute_error(iou, p)
        rmse = np.sqrt(mean_squared_error(iou, p))
        r2 = r2_score(iou, p)
        print(f"{label:14s} {sp:>9.3f} {pe:>8.3f} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f}")

    best = max(preds, key=lambda k: spearmanr(iou, preds[k]).correlation)
    p = preds[best]
    print(f"\nbest regressor: {best}")

    print(f"\nregress-then-threshold vs DIRECT classifier (accept=IoU>={IOU_ACCEPT}):")
    auc, prrej, acc = binary_from_score(iou, p, IOU_ACCEPT)
    print(f"  regress@0.90   ROC-AUC {auc:.3f}   PR-AUC(reject) {prrej:.3f}   accept {100*acc:.1f}%")
    print(f"  direct clf     ROC-AUC {DIRECT_CLASSIFIER_AUC:.3f}   (headline)")
    print(f"  delta          {auc - DIRECT_CLASSIFIER_AUC:+.3f}")

    print(f"\nsame single regression model, binary metrics at every threshold:")
    print(f"{'IoU cut':>8s} {'accept%':>9s} {'ROC-AUC':>8s} {'PR-AUC(rej)':>12s}")
    print("-" * 42)
    rows = [{"model": best, "iou_threshold": IOU_ACCEPT, "roc_auc_regress": auc,
             "pr_auc_reject": prrej, "direct_clf_auc": DIRECT_CLASSIFIER_AUC}]
    sweep = []
    for t in THRESHOLDS:
        a, pr, ac = binary_from_score(iou, p, t)
        print(f"{t:>8.2f} {100*ac:>8.1f}% {a:>8.3f} {pr:>12.3f}")
        sweep.append({"iou_threshold": t, "accept_rate": ac, "roc_auc": a, "pr_auc_reject": pr})

    pd.DataFrame(sweep).to_csv(os.path.join(OUT_DIR, "iou_regression_threshold_sweep.csv"), index=False)
    out = df[["subject", "organ", "iou"]].copy()
    for k, v in preds.items():
        out[f"pred_iou_{k}"] = v
    out.to_csv(os.path.join(OUT_DIR, "iou_regression_oof.csv"), index=False)
    print(f"\n  wrote iou_regression_oof.csv + iou_regression_threshold_sweep.csv")
    return p


# ---------------------------------------------------------------- Task 2

def task2_quantile_intervals(df, cols, groups):
    print("\n" + "=" * 74)
    print("2. PREDICTION INTERVALS (quantile regression)")
    print("=" * 74)
    iou = df["iou"].to_numpy()
    q10 = oof_quantile(df, cols, groups, 0.10)
    q50 = oof_quantile(df, cols, groups, 0.50)
    q90 = oof_quantile(df, cols, groups, 0.90)
    q10, q50, q90 = (np.clip(x, 0, 1) for x in (q10, q50, q90))
    # enforce monotone quantiles per row (quantile crossing is possible across separate fits)
    lo = np.minimum(q10, q90)
    hi = np.maximum(q10, q90)
    width = hi - lo

    inside = (iou >= lo) & (iou <= hi)
    coverage = inside.mean()
    print(f"\n  nominal coverage        80.0%")
    print(f"  empirical coverage      {100*coverage:.1f}%   (gap {100*(coverage-0.80):+.1f} pts)")
    print(f"  mean interval width     {width.mean():.3f}")
    print(f"  median interval width   {np.median(width):.3f}")

    abs_err = np.abs(iou - q50)
    sp = spearmanr(width, abs_err).correlation
    print(f"\n  Spearman(width, |true - q50|) = {sp:.3f}")
    print("  positive => wider intervals really do flag harder cases (usable 2nd signal)")

    fam = np.array([family_of(o) for o in df["organ"]])
    rows = []
    for f in sorted(set(fam)):
        sel = fam == f
        rows.append({"family": f, "n": int(sel.sum()), "coverage": inside[sel].mean(),
                     "mean_width": width[sel].mean()})
    fam_df = pd.DataFrame(rows).sort_values("mean_width")
    print(f"\n  per-family interval width & coverage:")
    print(f"  {'family':18s} {'n':>5s} {'coverage':>9s} {'mean_width':>11s}")
    for r in fam_df.itertuples(index=False):
        print(f"  {r.family:18s} {r.n:>5d} {100*r.coverage:>8.1f}% {r.mean_width:>11.3f}")

    out = df[["subject", "organ", "iou"]].copy()
    out["q10"], out["q50"], out["q90"] = lo, q50, hi
    out["width"], out["inside80"] = width, inside.astype(int)
    out.to_csv(os.path.join(OUT_DIR, "quantile_intervals_oof.csv"), index=False)
    fam_df.to_csv(os.path.join(OUT_DIR, "quantile_interval_by_family.csv"), index=False)
    print(f"\n  wrote quantile_intervals_oof.csv + quantile_interval_by_family.csv")


# ---------------------------------------------------------------- Task 2b (conformal)
#
# Conformalized Quantile Regression (CQR):
#   Romano, Patterson & Candes, "Conformalized Quantile Regression",
#   NeurIPS 2019 (arXiv:1905.03222).
# CQR keeps the ADAPTIVE (input-varying) width of the quantile regressor and only
# shifts each endpoint by a single scalar Q learned on a held-out calibration set - as
# opposed to a fixed-width conformal band on the point prediction, which would discard
# the adaptivity we measured (width-vs-error Spearman 0.475).
#
# Per the paper (Algorithm 1), with lower/upper quantile regressors q_lo, q_hi fit at
# alpha/2 and 1-alpha/2 on the proper-training set:
#   conformity score   E_i = max( q_lo(x_i) - y_i ,  y_i - q_hi(x_i) )    on calibration
#   Q = the ceil((n+1)(1-alpha))/n empirical quantile of {E_i}  (the k-th smallest,
#       k = ceil((n+1)(1-alpha)))
#   interval(x) = [ q_lo(x) - Q ,  q_hi(x) + Q ]
# A positive E means the point fell outside [q_lo, q_hi] (widen); a negative E means it
# was comfortably inside (Q<0 shrinks the interval). Coverage is guaranteed MARGINALLY
# over exchangeable calibration/test points; see the grouped-split caveat below.


def fit_quantile_model(train_df, cols, q):
    m = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("reg", HistGradientBoostingRegressor(loss="quantile", quantile=q,
                          max_iter=400, random_state=SEED))])
    m.fit(train_df[cols], train_df["iou"].to_numpy())
    return m


def conformal_Q(E, alpha):
    """The ceil((n+1)(1-alpha))-th smallest conformity score (CQR's finite-sample Q).

    If ceil((n+1)(1-alpha)) > n the quantile is undefined (too few calibration points
    for this alpha) - fall back to max(E), the most conservative finite choice.
    """
    E = np.asarray(E, dtype=float)
    n = len(E)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(np.max(E))
    return float(np.sort(E)[k - 1])


def cqr_cross_conformal(df, cols, groups, alpha, cal_frac, seed, mondrian=False):
    """Cross-conformal CQR: outer GroupKFold on subject gives every mask an out-of-fold
    interval; within each outer-training set we split BY SUBJECT again into proper-train
    (fit q_lo/q_hi) and calibration (compute Q). mondrian=True calibrates a separate Q
    per organ family, falling back to the global Q when a family has < MONDRIAN_MIN_CAL
    calibration masks.

    Returns (raw_lo, raw_hi, lo, hi, cal_sizes) - raw_* are the un-conformalised quantile
    predictions (the 'before'); lo/hi are the conformalised interval; cal_sizes maps
    family -> list of per-fold calibration counts (mondrian only).
    """
    n = len(df)
    raw_lo = np.full(n, np.nan); raw_hi = np.full(n, np.nan)
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    cal_sizes = defaultdict(list)
    q_lo_level, q_hi_level = alpha / 2, 1 - alpha / 2
    y = df["iou"].to_numpy()
    fam_all = np.array([family_of(o) for o in df["organ"]])

    outer = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    for tr_idx, te_idx in outer.split(df, y, groups):
        tr_df = df.iloc[tr_idx]
        tr_groups = groups[tr_idx]
        # proper-train / calibration split, BY SUBJECT (exchangeability across masks is
        # broken within a subject, so a row split would invalidate the guarantee)
        gss = GroupShuffleSplit(n_splits=1, test_size=cal_frac, random_state=seed)
        pt_rel, cal_rel = next(gss.split(tr_df, tr_df["iou"], tr_groups))
        pt_df, cal_df = tr_df.iloc[pt_rel], tr_df.iloc[cal_rel]

        m_lo = fit_quantile_model(pt_df, cols, q_lo_level)
        m_hi = fit_quantile_model(pt_df, cols, q_hi_level)

        cal_lo = m_lo.predict(cal_df[cols]); cal_hi = m_hi.predict(cal_df[cols])
        cal_y = cal_df["iou"].to_numpy()
        E = np.maximum(cal_lo - cal_y, cal_y - cal_hi)          # NOT clipped to [0,1]

        te_lo = m_lo.predict(df.iloc[te_idx][cols])
        te_hi = m_hi.predict(df.iloc[te_idx][cols])
        raw_lo[te_idx] = te_lo; raw_hi[te_idx] = te_hi

        if not mondrian:
            Q = conformal_Q(E, alpha)
            lo[te_idx] = te_lo - Q; hi[te_idx] = te_hi + Q
        else:
            Q_global = conformal_Q(E, alpha)
            floor = mondrian_floor(alpha)
            cal_fam = np.array([family_of(o) for o in cal_df["organ"]])
            te_fam = fam_all[te_idx]
            for fam in np.unique(te_fam):
                m_cal = cal_fam == fam
                n_cal = int(m_cal.sum())
                cal_sizes[fam].append(n_cal)
                Qf = conformal_Q(E[m_cal], alpha) if n_cal >= floor else Q_global
                sel = te_idx[te_fam == fam]
                lo[sel] = te_lo[te_fam == fam] - Qf
                hi[sel] = te_hi[te_fam == fam] + Qf
    return raw_lo, raw_hi, lo, hi, cal_sizes


def interval_report(y, lo, hi, fam, abs_err, alpha, label, min_cal=None, cal_sizes=None):
    """Print coverage / width / adaptivity for one interval set; return a per-family df."""
    inside = (y >= lo) & (y <= hi)
    width = np.clip(hi - lo, 0, None)
    cov, mw, mdw = inside.mean(), width.mean(), np.median(width)
    sp = spearmanr(width, abs_err).correlation
    nominal = 1 - alpha
    print(f"\n  [{label}]  nominal {100*nominal:.0f}%")
    print(f"    empirical coverage   {100*cov:.1f}%   (gap {100*(cov-nominal):+.1f} pts)")
    print(f"    mean / median width  {mw:.3f} / {mdw:.3f}")
    print(f"    Spearman(width,|y-q50|) = {sp:.3f}   (adaptivity retained if > 0)")
    rows = []
    for f in sorted(set(fam)):
        s = fam == f
        row = {"family": f, "n": int(s.sum()), "coverage": inside[s].mean(),
               "mean_width": width[s].mean()}
        if cal_sizes is not None:
            avg_cal = np.mean(cal_sizes.get(f, [0])) if cal_sizes.get(f) else 0
            row["avg_cal_n"] = round(avg_cal, 1)
            row["reliable"] = int(avg_cal >= (min_cal or 0))
        rows.append(row)
    return cov, mw, mdw, sp, inside, pd.DataFrame(rows)


def task2b_conformal(df, cols, groups):
    print("\n" + "=" * 74)
    print("2b. CONFORMALIZED QUANTILE REGRESSION (CQR)")
    print("=" * 74)
    print("  train/calibration split is BY SUBJECT; the coverage guarantee is MARGINAL")
    print("  over masks, and within-subject clustering (~70 masks/subject) is a caveat.")

    y = df["iou"].to_numpy()
    fam = np.array([family_of(o) for o in df["organ"]])
    q50 = np.clip(oof_quantile(df, cols, groups, 0.50), 0, 1)   # for the adaptivity metric
    abs_err = np.abs(y - q50)

    summary = []
    for alpha in CONF_ALPHAS:
        print("\n" + "-" * 70)
        print(f"alpha = {alpha}  (nominal {100*(1-alpha):.0f}% interval)")
        print("-" * 70)

        raw_lo, raw_hi, glo, ghi, _ = cqr_cross_conformal(df, cols, groups, alpha, CONF_CAL_FRAC, SEED)
        _, _, mlo, mhi, cal_sizes = cqr_cross_conformal(df, cols, groups, alpha, CONF_CAL_FRAC, SEED, mondrian=True)

        # clip conformity-corrected intervals to IoU's valid range [0,1] (scores were
        # computed BEFORE clipping); y in [0,1] so clipping cannot drop a covered point.
        variants = {
            "uncalibrated (raw quantiles)": (raw_lo, raw_hi, None),
            "CQR global":                   (glo, ghi, None),
            "CQR global (clipped [0,1])":   (np.clip(glo, 0, 1), np.clip(ghi, 0, 1), None),
            "Mondrian (per-family)":        (mlo, mhi, cal_sizes),
            "Mondrian (clipped [0,1])":     (np.clip(mlo, 0, 1), np.clip(mhi, 0, 1), cal_sizes),
        }
        floor = mondrian_floor(alpha)
        fam_tables = {}
        for name, (lo_, hi_, cs) in variants.items():
            cov, mw, mdw, sp, _, ftab = interval_report(
                y, lo_, hi_, fam, abs_err, alpha, name,
                min_cal=floor, cal_sizes=cs)
            fam_tables[name] = ftab
            summary.append({"alpha": alpha, "nominal": 1 - alpha, "variant": name,
                            "coverage": cov, "mean_width": mw, "median_width": mdw,
                            "spearman_width_err": sp})

        # per-family coverage: global vs Mondrian, with calibration-set sizes
        g = fam_tables["CQR global (clipped [0,1])"].set_index("family")
        m = fam_tables["Mondrian (clipped [0,1])"].set_index("family")
        print(f"\n  per-family coverage (target {100*(1-alpha):.0f}%): global vs Mondrian")
        print(f"  {'family':16s} {'n':>5s} {'avg_cal':>8s} {'globalCov':>10s} {'MondrCov':>9s} {'MondrW':>7s} {'ok?':>4s}")
        for famname in g.index:
            avg_cal = m.loc[famname, "avg_cal_n"]
            ok = "yes" if avg_cal >= floor else "SMALL"
            print(f"  {famname:16s} {int(g.loc[famname,'n']):>5d} {avg_cal:>8.1f} "
                  f"{100*g.loc[famname,'coverage']:>9.1f}% {100*m.loc[famname,'coverage']:>8.1f}% "
                  f"{m.loc[famname,'mean_width']:>7.3f} {ok:>4s}")

        if alpha == 0.2:
            out = df[["subject", "organ", "iou"]].copy()
            out["fam"] = fam
            out["raw_lo"], out["raw_hi"] = np.clip(raw_lo, 0, 1), np.clip(raw_hi, 0, 1)
            out["cqr_lo"], out["cqr_hi"] = np.clip(glo, 0, 1), np.clip(ghi, 0, 1)
            out["mondrian_lo"], out["mondrian_hi"] = np.clip(mlo, 0, 1), np.clip(mhi, 0, 1)
            out["q50"] = q50
            out.to_csv(os.path.join(OUT_DIR, "conformal_intervals_oof.csv"), index=False)
            m.reset_index().to_csv(os.path.join(OUT_DIR, "conformal_by_family.csv"), index=False)

    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "conformal_summary.csv"), index=False)
    print(f"\n  wrote conformal_intervals_oof.csv + conformal_by_family.csv + conformal_summary.csv")


# ---------------------------------------------------------------- Task 3

def task3_volume_label(df_all, cols, groups_all, df, groups):
    print("\n" + "=" * 74)
    print("3. VOLUME-ERROR LABEL")
    print("=" * 74)

    # sanity: relative volume error via voxels == via mm3 (voxel-volume factor cancels)
    s = df[(df.gt_vox > 0) & (df.num_voxels > 0)].head(5).copy()
    s["voxvol"] = s.volume_mm3 / s.num_voxels
    s["relerr_vox"] = (s.pred_vox - s.gt_vox) / s.gt_vox
    s["relerr_mm3"] = (s.volume_mm3 - s.gt_vox * s.voxvol) / (s.gt_vox * s.voxvol)
    print("\n  sanity check - relative volume error, voxel form vs mm3 form:")
    for r in s.itertuples(index=False):
        print(f"    {r.organ:22s} vox {r.relerr_vox:+.4f}   mm3 {r.relerr_mm3:+.4f}")
    print("    (identical => voxel-volume factor cancels, no spacing lookup needed)")

    for tau in VOL_TAUS:
        print(f"\n  --- vol_bad = |pred_vox - gt_vox|/gt_vox > {tau:.2f} ---")
        d = df.copy()
        d["vol_err"] = (d.pred_vox - d.gt_vox).abs() / d.gt_vox
        d["vol_bad"] = (d.vol_err > tau).astype(int)
        d["iou_reject"] = (d.iou < IOU_ACCEPT).astype(int)
        print(f"  prevalence: {100*d.vol_bad.mean():.1f}% vol_bad   ({int(d.vol_bad.sum())}/{len(d)} masks)")

        g = d["subject"].to_numpy()
        p = oof_proba(d, cols, "vol_bad", g)
        y = d.vol_bad.to_numpy()
        auc = roc_auc_score(y, p)
        pr = average_precision_score(y, p)
        ece, _, _ = ece_mce(y.astype(float), p)
        print(f"  classifier: ROC-AUC {auc:.3f}   PR-AUC(bad) {pr:.3f}   ECE {ece:.3f}")

        # agreement vs the IoU-0.90 label
        vb, ir = d.vol_bad.to_numpy(), d.iou_reject.to_numpy()
        agree = (vb == ir).mean()
        both = int(((vb == 1) & (ir == 1)).sum())
        vol_only = int(((vb == 1) & (ir == 0)).sum())
        iou_only = int(((vb == 0) & (ir == 1)).sum())
        neither = int(((vb == 0) & (ir == 0)).sum())
        print(f"  vs IoU<{IOU_ACCEPT} label: agreement {100*agree:.1f}%")
        print(f"    confusion (rows=vol_bad, cols=iou_reject):")
        print(f"                 iou_ok   iou_reject")
        print(f"      vol_ok     {neither:>6d}   {iou_only:>10d}")
        print(f"      vol_bad    {vol_only:>6d}   {both:>10d}")
        print(f"    vol_bad flags {vol_only} masks IoU-label misses; "
              f"IoU-label flags {iou_only} masks the volume check misses")

        if tau == 0.10:
            d.to_csv(os.path.join(OUT_DIR, "volume_label_oof.csv"), index=False)
            _volume_feature_ablation(d, cols, auc)
            _volume_only_miss(d)


def _volume_feature_ablation(d, cols, full_auc):
    """Leave-one-group-out on the volume label - does size become load-bearing?"""
    print(f"\n  --- feature-group ablation on the VOLUME label (full AUC {full_auc:.3f}) ---")
    print("  (IoU-label ablation: dropping size cost only 0.010 AUC. Does it flip here?)")
    g = d["subject"].to_numpy()
    groups_present = {gr: [c for c in cs if c in cols] for gr, cs in FEATURE_GROUPS.items()}
    rows = []
    print(f"  {'config':22s} {'AUC':>7s} {'delta':>8s}")
    print("  leave-one-group-out (marginal loss when this group is removed):")
    for gr, cs in groups_present.items():
        if not cs:
            continue
        sub = [c for c in cols if c not in cs]
        p = oof_proba(d, sub, "vol_bad", g)
        auc = roc_auc_score(d.vol_bad, p)
        rows.append({"kind": "drop_group", "group": gr, "auc": auc, "delta_auc": auc - full_auc})
    for r in sorted([x for x in rows if x["kind"] == "drop_group"], key=lambda x: x["delta_auc"]):
        print(f"    without {r['group']:12s} {r['auc']:>7.3f} {r['delta_auc']:>+8.3f}")
    print("  only-one-group (how much this group carries alone):")
    for gr, cs in groups_present.items():
        if not cs:
            continue
        p = oof_proba(d, cs, "vol_bad", g)
        auc = roc_auc_score(d.vol_bad, p)
        rows.append({"kind": "only_group", "group": gr, "auc": auc, "delta_auc": auc - full_auc})
    for r in sorted([x for x in rows if x["kind"] == "only_group"], key=lambda x: -x["auc"]):
        print(f"    only {r['group']:15s} {r['auc']:>7.3f} {r['delta_auc']:>+8.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "volume_label_ablation.csv"), index=False)


# ---------------------------------------------------------------- Task 4

# Volumetric clinical thresholds with a citable, per-organ, sex-independent fixed cutoff.
# Only structures meeting ALL of those criteria are kept (see DROPPED_ORGANS for the rest).
CLINICAL_CUTOFFS = {
    # Upper limit of normal splenic volume = 314.5 cm3 (Prassopoulos et al.,
    # "Determination of normal splenic volume on computed tomography in relation to age,
    # gender and body habitus", Eur Radiol 1997;7(2):246-248, PMID 9038125; mean 214.6,
    # range 107.2-314.5 mL). Reaffirmed by recent CT-volumetry work (~314 mL).
    "spleen": {"cutoff_ml": 314.5, "condition": "splenomegaly",
               "direction": "greater",   # volume ABOVE cutoff = disease
               "cite": "Prassopoulos 1997, Eur Radiol 7(2):246-248 (PMID 9038125)"},
}

# Organs deliberately excluded from the decision-flip analysis for lack of an applicable
# citable cutoff (per the 'no solid citation -> drop the organ' rule):
DROPPED_ORGANS = {
    "liver": "hepatomegaly threshold is weight-based (14.0*kg + 979 mL; Radiology 2021, "
             "PMC8805660) - no patient weight in the CSV, so it cannot be applied.",
    "kidney": "low-kidney-volume cutoff is for TOTAL (both kidneys) and sex-specific "
              "(337.5 mL men / 308.8 mL women; PMC10997556) - we segment L/R separately "
              "and have no sex column, so no sex-independent per-kidney cutoff applies.",
}

RELIABILITY_OPS = [0.80, 0.85, 0.90]   # predicted-IoU operating points for the filter


def task4_clinical_flips(df, pred_iou):
    print("\n" + "=" * 74)
    print("4. CLINICAL DECISION FLIPS")
    print("=" * 74)
    print("\n  dropped organs (no applicable citable volumetric cutoff):")
    for o, why in DROPPED_ORGANS.items():
        print(f"    {o}: {why}")

    d = df.copy()
    d["pred_iou"] = pred_iou
    d["voxvol"] = d.volume_mm3 / d.num_voxels
    d["gt_ml"] = d.gt_vox * d.voxvol / 1000
    d["pred_ml"] = d.volume_mm3 / 1000

    all_rows = []
    for organ, spec in CLINICAL_CUTOFFS.items():
        o = d[(d.organ == organ) & (d.gt_vox > 0)].copy()
        if o.empty:
            print(f"\n  {organ}: no rows in data, skipping")
            continue
        cut = spec["cutoff_ml"]
        o["gt_disease"] = o.gt_ml > cut
        o["pred_disease"] = o.pred_ml > cut
        o["flip"] = o.gt_disease != o.pred_disease
        n, nflip = len(o), int(o.flip.sum())
        print(f"\n  --- {organ} ({spec['condition']}, cutoff {cut} mL; {spec['cite']}) ---")
        print(f"  {n} masks | GT diseased {int(o.gt_disease.sum())} | "
              f"pred diseased {int(o.pred_disease.sum())} | decision flips {nflip} "
              f"({100*o.flip.mean():.1f}%)")

        if nflip == 0:
            closest = (o.pred_ml - cut).abs().min()
            print(f"  no flips: the mask never crosses the {cut} mL line the wrong way "
                  f"(closest predicted volume sits {closest:.0f} mL from the cutoff).")
            print(f"  -> nothing for the reliability filter to remove; TS {organ} volumetry "
                  f"is decision-robust here.")
        else:
            print(f"  {'op (predIoU>=)':>15s} {'review%':>9s} {'flips kept':>11s} "
                  f"{'flips removed':>14s} {'removed%':>9s}")
            for op in RELIABILITY_OPS:
                accept = o.pred_iou >= op
                flips_removed = int((o.flip & ~accept).sum())   # flagged for review
                flips_kept = int((o.flip & accept).sum())       # auto-accepted, still wrong
                review = 100 * (~accept).mean()
                removed_pct = 100 * flips_removed / nflip
                print(f"  {op:>15.2f} {review:>8.1f}% {flips_kept:>11d} "
                      f"{flips_removed:>14d} {removed_pct:>8.1f}%")
                print(f"    -> filtering removes {removed_pct:.0f}% of decision-flipping "
                      f"volumetric errors at a cost of {review:.0f}% review burden.")
        o["organ_cutoff_ml"] = cut
        all_rows.append(o[["subject", "organ", "gt_ml", "pred_ml", "gt_disease",
                           "pred_disease", "flip", "iou", "pred_iou", "organ_cutoff_ml"]])

    if all_rows:
        pd.concat(all_rows).to_csv(os.path.join(OUT_DIR, "clinical_decision_flips.csv"), index=False)
        print(f"\n  wrote clinical_decision_flips.csv")


def _volume_only_miss(d):
    """Masks a volume-only check would clear but IoU flags: correctly sized, displaced."""
    missed = ((d.vol_err < 0.10) & (d.iou < IOU_ACCEPT)).sum()
    small_err = (d.vol_err < 0.10).sum()
    print(f"\n  CAVEAT - volume error is necessary but not sufficient:")
    print(f"    {int(missed)} masks have vol_err<0.10 but IoU<0.90 "
          f"({100*missed/len(d):.1f}% of all masks, {100*missed/small_err:.1f}% of low-vol-err masks)")
    print("    - correctly sized but spatially displaced; a volume-only check clears them.")


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df_all = normalize_columns(pd.read_csv(CSV))
    df_all["family"] = [family_of(o) for o in df_all["organ"]]

    n_empty = int((df_all.is_empty == 1).sum())
    df = df_all[df_all.is_empty == 0].copy().reset_index(drop=True)
    print(f"{len(df_all)} masks total | dropped {n_empty} is_empty==1 rows "
          f"(trivially caught by an 'empty prediction' rule) | {len(df)} modelled")
    print(f"{df.subject.nunique()} subjects | {df.organ.nunique()} organs | "
          f"{df.family.nunique()} families")

    cols = feature_columns(df)
    print(f"{len(cols)} ground-truth-free features (leak guard passed)\n")
    groups = df["subject"].to_numpy()

    pred_iou = None
    if RUN_IOU_REG:
        pred_iou = task1_iou_regression(df, cols, groups)
    if RUN_QUANTILE:
        task2_quantile_intervals(df, cols, groups)
    if RUN_CONFORMAL:
        task2b_conformal(df, cols, groups)
    if RUN_VOLUME:
        # volume label needs gt_vox>0; drop false positives (gt_vox==0) for this task only
        dv = df[df.gt_vox > 0].copy().reset_index(drop=True)
        print(f"\n(volume tasks: dropped {len(df)-len(dv)} gt_vox==0 false-positive rows)")
        task3_volume_label(df_all, cols, groups, dv, dv["subject"].to_numpy())
    if RUN_CLINICAL:
        if pred_iou is None:
            pred_iou = oof_regress(df, cols, "iou", groups, "rf")
            pred_iou = np.clip(pred_iou, 0, 1)
        task4_clinical_flips(df, pred_iou)

    print(f"\nall CSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
