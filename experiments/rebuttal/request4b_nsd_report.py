"""
Request 4 reporting: full CT dev NSD analysis from request4_nsd_permask_dev.csv.
  - overall distribution per tolerance
  - highlighted structures (best/worst by NSD; the 6 below-chance within-structure-AUC ones)
  - IoU-vs-NSD confusion table (accept = IoU>=0.90 vs surface-accept = NSD>=0.95) per tolerance
  - disagreement vs structure volume (quartiles of reference voxel count)
  - Spearman and model-score AUC against NSD (from the summary)
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from experiments.pipeline.ablations import family_of

OUT = "rebuttal_results/ct"
perm = pd.read_csv(f"{OUT}/request4_nsd_permask_dev.csv")
oof = pd.read_csv("experiments/paper_final/ct/primary_model_oof.csv")[["subject", "organ", "randomforest_[raw]"]]
m = perm.merge(oof, on=["subject", "organ"], how="left").rename(columns={"randomforest_[raw]": "model_score"})
m["family"] = [family_of(o) for o in m["organ"]]
m["accept_iou"] = (m["iou"] >= 0.90).astype(int)
VOXVOL = 1.5 ** 3
m["ref_vol_mm3"] = m["gt_vox"] * VOXVOL
TAUS = [1, 2, 3]

# 1. overall distribution
dist = []
for t in TAUS:
    v = m[f"nsd_{t}mm"].dropna().to_numpy()
    dist.append({"tolerance_mm": t, "n": len(v), "mean": round(v.mean(), 4), "median": round(np.median(v), 4),
                 "std": round(v.std(), 4), "p05": round(np.percentile(v, 5), 4), "p25": round(np.percentile(v, 25), 4),
                 "p75": round(np.percentile(v, 75), 4), "p95": round(np.percentile(v, 95), 4),
                 "min": round(v.min(), 4), "frac_ge_0.95": round((v >= 0.95).mean(), 4)})
dist = pd.DataFrame(dist)
print("=== 1. CT dev NSD overall distribution ===")
print(dist.to_string(index=False))

# 2. highlighted structures by NSD@2mm
by_struct = m.groupby("organ").agg(n=("organ", "size"), nsd1=("nsd_1mm", "mean"),
                                   nsd2=("nsd_2mm", "mean"), nsd3=("nsd_3mm", "mean"),
                                   family=("family", "first")).reset_index()
by_struct[["nsd1", "nsd2", "nsd3"]] = by_struct[["nsd1", "nsd2", "nsd3"]].round(4)
by_struct.sort_values("nsd2").to_csv(f"{OUT}/request4_nsd_by_structure_dev.csv", index=False)
print("\n=== 2. worst 8 structures by mean NSD@2mm ===")
print(by_struct.sort_values("nsd2").head(8).to_string(index=False))
print("--- NSD@2mm for the 6 below-chance within-structure-AUC structures ---")
below = ["duodenum", "brachiocephalic_trunk", "gallbladder", "vertebrae_T1", "superior_vena_cava", "common_carotid_artery_left"]
print(by_struct[by_struct.organ.isin(below)].sort_values("nsd2").to_string(index=False))

# 3. IoU-vs-NSD confusion (surface-accept = NSD>=0.95)
conf_rows = []
for t in TAUS:
    v = m[f"nsd_{t}mm"].to_numpy()
    ok = ~np.isnan(v)
    ai = m["accept_iou"].to_numpy()[ok]
    an = (v[ok] >= 0.95).astype(int)
    tp = int(((ai == 1) & (an == 1)).sum()); fn = int(((ai == 1) & (an == 0)).sum())
    fp = int(((ai == 0) & (an == 1)).sum()); tn = int(((ai == 0) & (an == 0)).sum())
    conf_rows.append({"tolerance_mm": t, "n": int(ok.sum()),
                      "iouACC_nsdACC": tp, "iouACC_nsdREJ": fn, "iouREJ_nsdACC": fp, "iouREJ_nsdREJ": tn,
                      "agreement": round((tp + tn) / ok.sum(), 4)})
conf = pd.DataFrame(conf_rows)
conf.to_csv(f"{OUT}/request4_nsd_iou_confusion_dev.csv", index=False)
print("\n=== 3. IoU(>=0.90) vs surface-accept(NSD>=0.95) confusion ===")
print(conf.to_string(index=False))

# 4. disagreement vs reference volume (quartiles), using NSD@2mm surface-accept
bp = m[(m.gt_vox >= 20)].copy()                        # both-present (real surface pair)
bp["disagree"] = (bp["accept_iou"] != (bp["nsd_2mm"] >= 0.95).astype(int)).astype(int)
bp["vol_q"] = pd.qcut(bp["ref_vol_mm3"], 4, labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"])
vol = bp.groupby("vol_q", observed=True).agg(n=("disagree", "size"),
        ref_vol_median_mm3=("ref_vol_mm3", "median"), mean_nsd2=("nsd_2mm", "mean"),
        disagreement_rate=("disagree", "mean")).reset_index()
vol[["ref_vol_median_mm3", "mean_nsd2", "disagreement_rate"]] = vol[["ref_vol_median_mm3", "mean_nsd2", "disagreement_rate"]].round(4)
vol.to_csv(f"{OUT}/request4_nsd_disagreement_by_volume_dev.csv", index=False)
print("\n=== 4. IoU-vs-NSD@2mm disagreement by reference-volume quartile ===")
print(vol.to_string(index=False))

# 5. Spearman / model-score AUC against NSD
sp_rows = []
for t in TAUS:
    v = m[f"nsd_{t}mm"].to_numpy(); ok = ~np.isnan(v)
    lab = (v[ok] >= 0.95).astype(int)
    sp_rows.append({"tolerance_mm": t,
                    "spearman_model_vs_nsd": round(spearmanr(m["model_score"][ok], v[ok]).correlation, 4),
                    "spearman_iou_vs_nsd": round(spearmanr(m["iou"][ok], v[ok]).correlation, 4),
                    "model_auc_for_nsd_ge_0.95": round(roc_auc_score(lab, m["model_score"].to_numpy()[ok]), 4)
                    if len(np.unique(lab)) == 2 else np.nan})
sp = pd.DataFrame(sp_rows)
print("\n=== 5. OOF score vs NSD (Spearman + AUC) ===")
print(sp.to_string(index=False))
dist.to_csv(f"{OUT}/request4_nsd_distribution_dev.csv", index=False)
