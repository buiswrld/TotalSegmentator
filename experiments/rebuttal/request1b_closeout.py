"""
Request 1 closeout (no new inference):
 - Paired bootstrap difference (primary model minus organ-rate baseline) ROC-AUC, with 95%
   subject-bootstrap CI, for MRI DEV and MRI TEST side by side.
 - Frozen MRI no-intensity secondary on test: same RF IoU regressor on the 14 non-HU
   features (INTENSITY_COLS dropped), refit on dev, scored on the 49-subject test.
All seeds 0, subject-grouped. MRI test is PREVIOUSLY EXPOSED (see CONTAMINATION_AUDIT.md).
"""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from experiments.pipeline.regression import normalize_columns, feature_columns, SEED, N_TREES, IOU_ACCEPT
from experiments.pipeline.paper_consolidate import INTENSITY_COLS

PF = "experiments/paper_final"
OUT = "rebuttal_results/mri"

dev = pd.read_csv(f"{PF}/mri/dataset_combined_metrics.csv"); dev = dev[dev.is_empty == 0].reset_index(drop=True)
test = normalize_columns(pd.read_csv("experiments/eval_runs/mr_full_run/metrics/test/combined_metrics.csv"))
test = test[test.is_empty == 0].reset_index(drop=True)
cols = feature_columns(dev)
y_dev = (dev["iou"].to_numpy() >= IOU_ACCEPT).astype(int)
y_test = (test["iou"].to_numpy() >= IOU_ACCEPT).astype(int)


def organ_rate_oof(df, y, groups):
    p = np.zeros(len(df)); lab = pd.Series(y, index=df.index)
    for tr, te in GroupKFold(5).split(df, y, groups):
        rates = lab.iloc[tr].groupby(df.iloc[tr]["organ"].values).mean()
        p[te] = df.iloc[te]["organ"].map(rates).fillna(y[tr].mean()).to_numpy()
    return p


def paired_diff_ci(y, s_model, s_base, subjects, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); uniq = np.unique(subjects); diffs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(subjects == s)[0] for s in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, s_model[idx]) - roc_auc_score(yy, s_base[idx]))
    point = roc_auc_score(y, s_model) - roc_auc_score(y, s_base)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---- DEV paired diff (frozen OOF model vs OOF organ-rate) ----
oof = pd.read_csv(f"{PF}/mri/iou_regression_oof.csv")
assert (oof["subject"].values == dev["subject"].values).all() and (oof["organ"].values == dev["organ"].values).all()
dev_model = oof["pred_iou_RandomForest"].to_numpy()
dev_base = organ_rate_oof(dev, y_dev, dev["subject"].to_numpy())
dp, dlo, dhi = paired_diff_ci(y_dev, dev_model, dev_base, dev["subject"].to_numpy())

# ---- TEST paired diff (refit frozen primary + baseline on dev, apply to test) ----
reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                ("reg", RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2, random_state=SEED, n_jobs=-1))])
reg.fit(dev[cols], dev["iou"].to_numpy())
test_model = np.clip(reg.predict(test[cols]), 0, 1)
dev_rates = pd.Series(y_dev, index=dev.index).groupby(dev["organ"].values).mean()
test_base = pd.Series(test["organ"].values).map(dev_rates).fillna(y_dev.mean()).to_numpy()
tp, tlo, thi = paired_diff_ci(y_test, test_model, test_base, test["subject"].to_numpy())

diff_tab = pd.DataFrame([
    {"split": "MRI_dev", "model_auc": round(roc_auc_score(y_dev, dev_model), 4),
     "baseline_auc": round(roc_auc_score(y_dev, dev_base), 4),
     "paired_diff": round(dp, 4), "diff_ci_lo": round(dlo, 4), "diff_ci_hi": round(dhi, 4)},
    {"split": "MRI_test_exposed", "model_auc": round(roc_auc_score(y_test, test_model), 4),
     "baseline_auc": round(roc_auc_score(y_test, test_base), 4),
     "paired_diff": round(tp, 4), "diff_ci_lo": round(tlo, 4), "diff_ci_hi": round(thi, 4)},
])
diff_tab.to_csv(f"{OUT}/request1_model_vs_baseline_paired.csv", index=False)
print("=== Request 1: model vs organ-rate baseline, paired (dev and test side by side) ===")
print(diff_tab.to_string(index=False))

# ---- no-intensity secondary on test ----
cols_ni = [c for c in cols if c not in INTENSITY_COLS]
print(f"\nno-intensity feature set: {len(cols_ni)} features ({len(cols)} minus {len(INTENSITY_COLS)} HU)")
reg_ni = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("reg", RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2, random_state=SEED, n_jobs=-1))])
reg_ni.fit(dev[cols_ni], dev["iou"].to_numpy())
p_ni = np.clip(reg_ni.predict(test[cols_ni]), 0, 1)


def boot_metric(y, score, subjects, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); uniq = np.unique(subjects); vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(subjects == s)[0] for s in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(fn(yy, score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


subj = test["subject"].to_numpy()
ni_rows = []
for metric, fn, sc in [("ROC_AUC", roc_auc_score, p_ni),
                       ("PR_AUC_accept", average_precision_score, p_ni),
                       ("PR_AUC_reject", lambda y, s: average_precision_score(1 - y, -s), p_ni)]:
    pt = fn(y_test, sc); lo, hi = boot_metric(y_test, sc, subj, fn)
    ni_rows.append({"model": "RF_IoU_regressor_no_intensity", "metric": metric,
                    "point": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
ni = pd.DataFrame(ni_rows)
ni.to_csv(f"{OUT}/request1_no_intensity_test.csv", index=False)
print("\n=== Request 1: MRI no-intensity secondary on TEST (frozen 14-feature regressor) ===")
print(ni.to_string(index=False))
