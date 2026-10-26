"""
Request 1 (MRI test only): refit the FROZEN MRI models on the 38 dev subjects and score
the existing 49-subject held-out MRI test table (mr_full_run). No new inference.

Frozen MRI primary = RandomForestRegressor(300 trees, min_samples_leaf=2, seed 0) on the
19 reference-free features, median-imputed; score = predicted IoU. Baseline = per-structure
accept rate learned on dev. accept = (IoU >= 0.90). Subject bootstrap 1000 draws, 95% pct.

MRI test is a PREVIOUSLY EXPOSED split (see rebuttal_results/CONTAMINATION_AUDIT.md) and is
in-sample for the segmenter. Cite the audit beside every number here.
"""
import os, sys, glob
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import nibabel as nib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

from experiments.pipeline.regression import normalize_columns, feature_columns, SEED, N_TREES, IOU_ACCEPT

PF = "experiments/paper_final"
TEST_CSV = "experiments/eval_runs/mr_full_run/metrics/test/combined_metrics.csv"
MRI_DS = os.environ["SSAT_MRI_DATASET"]
OUT = "ssat_outputs/rebuttal/mri"
os.makedirs(OUT, exist_ok=True)
UNSCORED6 = ["s0078", "s0150", "s0458", "s0480", "s0523", "s0580"]
MIN_VOX = 20

# ---------------------------------------------------------------- load + align
dev = pd.read_csv(f"{PF}/mri/dataset_combined_metrics.csv")
dev = dev[dev.is_empty == 0].reset_index(drop=True)
test = normalize_columns(pd.read_csv(TEST_CSV))
test = test[test.is_empty == 0].reset_index(drop=True)

cols = feature_columns(dev)                       # 19 frozen features
assert len(cols) == 19, len(cols)

y_dev = (dev["iou"].to_numpy() >= IOU_ACCEPT).astype(int)
y_test = (test["iou"].to_numpy() >= IOU_ACCEPT).astype(int)
subj_test = test["subject"].to_numpy()
print(f"dev non-empty masks  = {len(dev)} over {dev.subject.nunique()} subjects, accept rate {y_dev.mean()*100:.1f}%")
print(f"test non-empty masks = {len(test)} over {test.subject.nunique()} subjects, accept rate {y_test.mean()*100:.1f}%")

# ---------------------------------------------------------------- frozen primary: RF IoU regressor
reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                ("reg", RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2,
                                              random_state=SEED, n_jobs=-1))])
reg.fit(dev[cols], dev["iou"].to_numpy())
pred_iou = np.clip(reg.predict(test[cols]), 0, 1)

# ---------------------------------------------------------------- frozen baseline: per-structure accept rate (dev)
dev_rates = pd.Series(y_dev, index=dev.index).groupby(dev["organ"].values).mean()
base_score = pd.Series(test["organ"].values).map(dev_rates).fillna(y_dev.mean()).to_numpy()

# ---------------------------------------------------------------- metrics + subject bootstrap
def metrics(y, score):
    return {"ROC_AUC": roc_auc_score(y, score),
            "PR_AUC_accept": average_precision_score(y, score),
            "PR_AUC_reject": average_precision_score(1 - y, -score)}

def boot_ci(y, score, subjects, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    keep = {k: [] for k in ["ROC_AUC", "PR_AUC_accept", "PR_AUC_reject"]}
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(subjects == s)[0] for s in pick])
        yy, ss = y[idx], score[idx]
        if yy.min() == yy.max():
            continue
        keep["ROC_AUC"].append(roc_auc_score(yy, ss))
        keep["PR_AUC_accept"].append(average_precision_score(yy, ss))
        keep["PR_AUC_reject"].append(average_precision_score(1 - yy, -ss))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)) for k, v in keep.items()}

rows = []
for name, score in [("RF_IoU_regressor_primary", pred_iou), ("organ_rate_baseline", base_score)]:
    m = metrics(y_test, score); ci = boot_ci(y_test, score, subj_test)
    for metric in ["ROC_AUC", "PR_AUC_accept", "PR_AUC_reject"]:
        lo, hi, nb = ci[metric]
        rows.append({"model": name, "metric": metric, "point": round(m[metric], 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n_boot_valid": nb})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/request1_test_metrics.csv", index=False)
print("\n=== Request 1: MRI held-out test (49 subjects, PREVIOUSLY EXPOSED; see CONTAMINATION_AUDIT.md) ===")
print(res.to_string(index=False))

# ---------------------------------------------------------------- 6 unscored subjects: accept-rate bound
rowsb = []
n_eligible_6 = 0
for s in UNSCORED6:
    seg = sorted(glob.glob(os.path.join(MRI_DS, s, "segmentations", "*.nii.gz")))
    ge20 = 0
    for f in seg:
        arr = nib.load(f).get_fdata()
        if int((arr > 0).sum()) >= MIN_VOX:
            ge20 += 1
    n_eligible_6 += ge20
    rowsb.append({"subject": s, "ref_structures_ge20vox": ge20, "ref_files": len(seg)})
bnd = pd.DataFrame(rowsb)
bnd.to_csv(f"{OUT}/request1_unscored6_refcounts.csv", index=False)

n_accept_49 = int(y_test.sum())
n_modelled_49 = len(test)
accept_excl = n_accept_49 / n_modelled_49
accept_worst = n_accept_49 / (n_modelled_49 + n_eligible_6)
print("\n=== 6 unscored MRI test subjects: reference structures with >= 20 voxels ===")
print(bnd.to_string(index=False))
print(f"\naccept rate EXCLUDING the 6 (49 subj, {n_accept_49}/{n_modelled_49} modelled) = {accept_excl*100:.1f}%")
print(f"accept rate WORST CASE (6 subj' {n_eligible_6} eligible ref structures all failed) "
      f"= {n_accept_49}/{n_modelled_49 + n_eligible_6} = {accept_worst*100:.1f}%")
pd.DataFrame([{"n_accept_49": n_accept_49, "n_modelled_49": n_modelled_49,
               "n_eligible_ref_6": n_eligible_6, "accept_rate_excl_pct": round(accept_excl*100, 1),
               "accept_rate_worstcase_pct": round(accept_worst*100, 1)}]).to_csv(
    f"{OUT}/request1_accept_rate_bound.csv", index=False)
