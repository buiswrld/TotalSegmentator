"""
Request 4, Dice summary for CT dev, MRI dev, and MRI test, from the existing on-disk metric
tables (no mask loading). Dice is already stored per (subject, organ). Non-empty modelled
masks only (is_empty == 0), consistent with the rest of the paper.
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from experiments.pipeline.regression import normalize_columns

os.makedirs("rebuttal_results/ct", exist_ok=True)
os.makedirs("rebuttal_results/mri", exist_ok=True)

srcs = {
    "CT_dev": "experiments/paper_final/ct/dataset_combined_metrics.csv",
    "MRI_dev": "experiments/paper_final/mri/dataset_combined_metrics.csv",
    "MRI_test_exposed": "experiments/eval_runs/mr_full_run/metrics/test/combined_metrics.csv",
}
rows = []
for name, path in srcs.items():
    df = normalize_columns(pd.read_csv(path))
    df = df[df.is_empty == 0]
    d = df["dice"].to_numpy()
    rows.append({"split": name, "n_masks": len(df), "n_subjects": df["subject"].nunique(),
                 "dice_mean": round(float(np.mean(d)), 4), "dice_median": round(float(np.median(d)), 4),
                 "dice_std": round(float(np.std(d)), 4), "dice_p05": round(float(np.percentile(d, 5)), 4),
                 "dice_p95": round(float(np.percentile(d, 95)), 4),
                 "frac_dice_ge_0.90": round(float((d >= 0.90).mean()), 4)})
tab = pd.DataFrame(rows)
tab.to_csv("rebuttal_results/ct/request4_dice_summary.csv", index=False)
tab.to_csv("rebuttal_results/mri/request4_dice_summary.csv", index=False)
print("=== Request 4: Dice summary (non-empty modelled masks) ===")
print(tab.to_string(index=False))
