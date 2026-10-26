"""
Phase 0 reproduction check: regenerate the paper's development OOF numbers from the
code that will be frozen, and confirm they match the committed paper_final summaries to
3 decimals. DEV ONLY - no test data is touched.

Targets (from paper_final):
  CT  primary RF classifier (raw 19 descriptors) dev OOF AUC = 0.871 ; organ-rate = 0.761
  MRI primary RF IoU-regressor (with intensity)  dev OOF AUC = 0.919 ; organ-rate = 0.815
"""
import os, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pipeline.folds import StableGroupKFold as GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

PF = "experiments/paper_final"
LEAK = ["dice","iou","tp","fp","fn","gt_vox","status","accept"]
IDC  = ["subject","organ","orig_axcodes","pred_vox","family","label","vol_err","vol_bad","iou_reject"]

def feats(df, drop_empty_const=True):
    cols = [c for c in df.columns if c not in LEAK+IDC]
    if drop_empty_const:
        cols = [c for c in cols if df[c].nunique(dropna=False) > 1]   # drops constant is_empty
    return cols

def organ_rate_oof(df, y, groups):
    p = np.zeros(len(df)); lab = pd.Series(y, index=df.index)
    for tr, te in GroupKFold(5).split(df, y, groups):
        rates = lab.iloc[tr].groupby(df.iloc[tr]["organ"].values).mean()
        p[te] = df.iloc[te]["organ"].map(rates).fillna(y[tr].mean()).to_numpy()
    return roc_auc_score(y, p)

def oof_clf(df, cols, y, groups, n_trees):
    p = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, y, groups):
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2,
                               class_weight="balanced_subsample", random_state=0, n_jobs=-1))])
        m.fit(df.iloc[tr][cols], y[tr]); p[te] = m.predict_proba(df.iloc[te][cols])[:,1]
    return roc_auc_score(y, p)

def oof_reg(df, cols, y, groups, n_trees):
    p = np.zeros(len(df))
    iou = df["iou"].to_numpy()
    for tr, te in GroupKFold(5).split(df, y, groups):
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("reg", RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                               random_state=0, n_jobs=-1))])
        m.fit(df.iloc[tr][cols], iou[tr]); p[te] = np.clip(m.predict(df.iloc[te][cols]),0,1)
    return roc_auc_score(y, p)

print("="*64, "\nCT (dev = 57 val subjects)")
ct = pd.read_csv(PF+"/ct/dataset_combined_metrics.csv")
ct = ct[ct.is_empty==0].reset_index(drop=True)
y = (ct["iou"].to_numpy() >= 0.90).astype(int); g = ct["subject"].to_numpy()
cols = feats(ct, drop_empty_const=False); print(f"  features used: {len(cols)}")
# stored OOF (sanity)
oof = pd.read_csv(PF+"/ct/primary_model_oof.csv")
print(f"  stored-OOF  RF[raw] AUC = {roc_auc_score((oof['iou']>=0.9).astype(int), oof['randomforest_[raw]']):.4f}"
      f"   organ-rate = {roc_auc_score((oof['iou']>=0.9).astype(int), oof['organ-rate_baseline']):.4f}")
# refit from code (train.py config: 400 trees)
print(f"  refit-code  RF[raw] AUC = {oof_clf(ct, cols, y, g, 400):.4f}   organ-rate = {organ_rate_oof(ct, y, g):.4f}   (targets 0.871 / 0.761)")

print("="*64, "\nMRI (dev = 38 subjects)")
mr = pd.read_csv(PF+"/mri/dataset_combined_metrics.csv")
mr = mr[mr.is_empty==0].reset_index(drop=True)
y = (mr["iou"].to_numpy() >= 0.90).astype(int); g = mr["subject"].to_numpy()
cols = feats(mr); print(f"  features used: {len(cols)}")
oofm = pd.read_csv(PF+"/mri/iou_regression_oof.csv")
print(f"  stored-OOF  RF-reg AUC = {roc_auc_score((oofm['iou']>=0.9).astype(int), oofm['pred_iou_RandomForest']):.4f}")
print(f"  refit-code  RF-reg AUC = {oof_reg(mr, cols, y, g, 300):.4f}   organ-rate = {organ_rate_oof(mr, y, g):.4f}   (targets 0.919 / 0.815)")
