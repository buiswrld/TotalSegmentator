"""
Request 4: Normalized Surface Dice (NSD) at 1, 2, 3 mm on CT DEV only (predictions_ct_val vs
reference masks). No new inference. Each NSD is computed on the union bounding box plus a
5-voxel margin for speed; since the max tolerance (3 mm = 2 voxels at 1.5 mm) is well inside
that margin, the thresholded distances are exact.

DEVIATION (documented): the frozen config named DeepMind surface-distance or MONAI
compute_surface_dice. Neither is installed in the analysis env, and the time-forced plan
forbids environment changes, so NSD is computed here directly with scipy using the standard
symmetric surface-overlap definition:

  NSD@tau = ( |{pred surface voxels within tau mm of ref surface}|
            + |{ref surface voxels within tau mm of pred surface}| )
            / ( |pred surface voxels| + |ref surface voxels| )

Surface = mask minus its binary erosion. Distances via scipy distance_transform_edt with mm
sampling. This matches the NSD definition (Nikolov 2021); the only difference from the named
libraries is a voxel-surface rather than a marching-cubes surface. Results are related to the
frozen CT dev out-of-fold scores (primary_model_oof.csv, 400-tree randomforest_[raw]).
"""
import os, sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PRED = os.environ["SSAT_CT_PREDICTIONS"]
DS = os.environ["SSAT_CT_DATASET"]
OUT = "ssat_outputs/rebuttal/ct"
TAUS = (1.0, 2.0, 3.0)
MARGIN = 5
os.makedirs(OUT, exist_ok=True)

dev = pd.read_csv("experiments/paper_final/ct/dataset_combined_metrics.csv")
nm = dev[dev.is_empty == 0].reset_index(drop=True)


from experiments.rebuttal._nsd import surf, nsd


rows = []
t0 = time.time()
for i, row in nm.iterrows():
    s, o = row["subject"], row["organ"]
    pred_present = row["pred_vox"] >= 20
    gt_present = row["gt_vox"] >= 20
    rec = {"subject": s, "organ": o, "pred_vox": int(row["pred_vox"]), "gt_vox": int(row["gt_vox"]),
           "iou": row["iou"], "dice": row["dice"]}
    if pred_present and gt_present:
        pf = os.path.join(PRED, s, o + ".nii.gz")
        rf = os.path.join(DS, s, "segmentations", o + ".nii.gz")
        pi, ri = nib.load(pf), nib.load(rf)
        P = np.asanyarray(pi.dataobj) > 0
        R = np.asanyarray(ri.dataobj) > 0
        res = nsd(P, R, pi.header.get_zooms()[:3])
    else:
        res = {t: 0.0 for t in TAUS}  # exactly one side present -> surface overlap is 0
    for t in TAUS:
        rec[f"nsd_{int(t)}mm"] = round(res[t], 5) if res[t] == res[t] else np.nan
    rows.append(rec)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(nm)} masks  ({time.time()-t0:.0f}s)")

perm = pd.DataFrame(rows)
perm.to_csv(f"{OUT}/request4_nsd_permask_dev.csv", index=False)
print(f"\ncomputed {len(perm)} CT dev masks in {time.time()-t0:.0f}s")

# relate NSD to frozen CT dev OOF scores
oof = pd.read_csv("experiments/paper_final/ct/primary_model_oof.csv")[["subject", "organ", "randomforest_[raw]"]]
m = perm.merge(oof, on=["subject", "organ"], how="left").rename(columns={"randomforest_[raw]": "model_score"})
y_accept = (m["iou"].to_numpy() >= 0.90).astype(int)

summ = []
for t in TAUS:
    col = f"nsd_{int(t)}mm"
    v = m[col].to_numpy()
    ok = ~np.isnan(v)
    summ.append({"tolerance_mm": int(t), "n": int(ok.sum()),
                 "nsd_mean": round(float(np.nanmean(v)), 4), "nsd_median": round(float(np.nanmedian(v)), 4),
                 "frac_nsd_ge_0.95": round(float((v[ok] >= 0.95).mean()), 4),
                 "spearman_model_vs_nsd": round(float(spearmanr(m["model_score"][ok], v[ok]).correlation), 4),
                 "spearman_iou_vs_nsd": round(float(spearmanr(m["iou"][ok], v[ok]).correlation), 4),
                 "mean_nsd_accepted": round(float(np.nanmean(v[y_accept == 1])), 4),
                 "mean_nsd_rejected": round(float(np.nanmean(v[y_accept == 0])), 4),
                 "model_auc_for_nsd_ge_0.95": round(float(roc_auc_score((v[ok] >= 0.95).astype(int), m["model_score"].to_numpy()[ok])), 4)
                 if len(np.unique((v[ok] >= 0.95).astype(int))) == 2 else np.nan})
summ = pd.DataFrame(summ)
summ.to_csv(f"{OUT}/request4_nsd_summary_dev.csv", index=False)
print("\n=== Request 4: CT DEV NSD summary (vs frozen 400-tree dev OOF score) ===")
print(summ.to_string(index=False))
