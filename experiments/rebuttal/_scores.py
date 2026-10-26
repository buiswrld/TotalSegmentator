"""Shared per-mask score frames for rebuttal Requests 2 and 3 (on-disk outputs only).

- ct_dev_frame():  CT primary 400-tree randomforest_[raw] dev OOF (primary_model_oof.csv).
                   DEV OOF, never test. NOT within_organ_auc.csv (that was the 200-tree run).
- mri_test_frame(): refit frozen MRI regressor + organ-rate baseline on 38 dev, score the
                    existing 49-subject test table. MRI test is previously exposed (audit).
Both return columns: subject, organ, iou, accept, score_model, score_base
(score = predicted acceptability; higher means more likely accept; risk = 1 - score).
"""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from experiments.pipeline.regression import normalize_columns, feature_columns, SEED, N_TREES, IOU_ACCEPT

PF = "experiments/paper_final"
MRI_TEST_CSV = "experiments/eval_runs/mr_full_run/metrics/test/combined_metrics.csv"


def ct_dev_frame():
    o = pd.read_csv(f"{PF}/ct/primary_model_oof.csv")
    return pd.DataFrame({
        "subject": o["subject"], "organ": o["organ"], "iou": o["iou"],
        "accept": (o["iou"].to_numpy() >= IOU_ACCEPT).astype(int),
        "score_model": o["randomforest_[raw]"].to_numpy(),      # frozen 400-tree primary, DEV OOF
        "score_base": o["organ-rate_baseline"].to_numpy(),
    })


def mri_test_frame():
    dev = pd.read_csv(f"{PF}/mri/dataset_combined_metrics.csv")
    dev = dev[dev.is_empty == 0].reset_index(drop=True)
    test = normalize_columns(pd.read_csv(MRI_TEST_CSV))
    test = test[test.is_empty == 0].reset_index(drop=True)
    cols = feature_columns(dev)
    y_dev = (dev["iou"].to_numpy() >= IOU_ACCEPT).astype(int)

    reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("reg", RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2,
                                                  random_state=SEED, n_jobs=-1))])
    reg.fit(dev[cols], dev["iou"].to_numpy())
    pred_iou = np.clip(reg.predict(test[cols]), 0, 1)

    dev_rates = pd.Series(y_dev, index=dev.index).groupby(dev["organ"].values).mean()
    base = pd.Series(test["organ"].values).map(dev_rates).fillna(y_dev.mean()).to_numpy()
    return pd.DataFrame({
        "subject": test["subject"], "organ": test["organ"], "iou": test["iou"],
        "accept": (test["iou"].to_numpy() >= IOU_ACCEPT).astype(int),
        "score_model": pred_iou, "score_base": base,
    })
