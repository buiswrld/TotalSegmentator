"""
TotalSegmentator MRI evaluation harness.

For each subject in the chosen split it will:
  1. make sure predictions exist (optionally running TotalSegmentator if they don't),
  2. score every organ against the dataset's ground truth (Dice + failure flags),
  3. save an overlay PNG (prediction vs ground truth, three planes),
and then aggregate everything into two CSVs plus a console summary.

Run from anywhere once the paths below are set:  python report_tool.py
"""

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # batch-safe; no windows pop up
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import glob, os, csv, subprocess
from collections import defaultdict
from scipy.stats import spearmanr

# ============================ CONFIG: edit these ============================
DATASET    = r"C:\Users\ansar\Downloads\TotalsegmentatorMRI_dataset_v200"
PRED_ROOT  = r"C:\Users\ansar\Algoverse\predictions"   # one subfolder per subject
REPORT     = r"C:\Users\ansar\Algoverse\report"        # outputs go here

SPLIT      = "test"     # which meta.csv split to evaluate; set None to use every subject
LIMIT      = 5          # cap number of subjects (good for a first trial); None = all
RUN_TS     = True       # if a subject has no predictions yet, run TotalSegmentator
TS_EXTRA   = ["-ta", "total_mr"]   # add "-f" here for the faster 3mm model on CPU

MIN_VOXELS = 20         # masks below this count are treated as absent (kills edge slivers)
DICE_FLAG  = 0.50       # Dice below this (when both present) is flagged low
SAVE_OVERLAYS = True

# Expected per-organ Dice for the TotalSegmentator MRI model (total_mr task).
# Source: wasserth/TotalSegmentator, resources/results_all_classes_mr.json
# (commit on GitHub repo; paper: Akinci D'Antonoli et al., Radiology 2025,
#  doi:10.1148/radiol.241613, arXiv:2405.19492 — overall mean Dice 0.839
#  on 55-subject internal test set).
EXPECTED_DICE = {
    "spleen": 0.927, "kidney_right": 0.932, "kidney_left": 0.944,
    "gallbladder": 0.903, "liver": 0.960, "stomach": 0.913,
    "pancreas": 0.700, "adrenal_gland_right": 0.721, "adrenal_gland_left": 0.733,
    "lung_left": 0.968, "lung_right": 0.979, "esophagus": 0.830,
    "small_bowel": 0.852, "duodenum": 0.772, "colon": 0.875,
    "urinary_bladder": 0.802, "prostate": 0.743, "sacrum": 0.870,
    "vertebrae": 0.918, "intervertebral_discs": 0.871, "spinal_cord": 0.937,
    "heart": 0.933, "aorta": 0.932, "inferior_vena_cava": 0.901,
    "portal_vein_and_splenic_vein": 0.766, "iliac_artery_left": 0.804,
    "iliac_artery_right": 0.716, "iliac_vena_left": 0.694,
    "iliac_vena_right": 0.598, "humerus_left": 0.874, "humerus_right": 0.827,
    "scapula_left": 0.761, "scapula_right": 0.728, "clavicula_left": 0.733,
    "clavicula_right": 0.803, "femur_left": 0.936, "femur_right": 0.960,
    "hip_left": 0.905, "hip_right": 0.906, "gluteus_maximus_left": 0.884,
    "gluteus_maximus_right": 0.924, "gluteus_medius_left": 0.879,
    "gluteus_medius_right": 0.935, "gluteus_minimus_left": 0.915,
    "gluteus_minimus_right": 0.868, "autochthon_left": 0.962,
    "autochthon_right": 0.952, "iliopsoas_left": 0.916, "iliopsoas_right": 0.906,
    "brain": 0.984,
}
# ===========================================================================


def find_columns(header):
    lower = [h.lower() for h in header]
    id_col = next((header[lower.index(c)] for c in
                   ("image_id", "subject", "id", "image", "name") if c in lower), None)
    split_col = next((header[lower.index(c)] for c in
                      ("split", "partition", "subset", "set", "fold") if c in lower), None)
    return id_col, split_col


def get_subjects(dataset, split, limit):
    """Subject ids from meta.csv filtered by split; falls back to all sXXXX folders."""
    meta = os.path.join(dataset, "meta.csv")
    subjects = []
    if split and os.path.exists(meta):
        with open(meta, newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096); fh.seek(0)
            delim = ";" if sample.count(";") > sample.count(",") else ","
            reader = csv.DictReader(fh, delimiter=delim)
            id_col, split_col = find_columns(reader.fieldnames or [])
            if id_col and split_col:
                subjects = [row[id_col].strip() for row in reader
                            if row.get(split_col, "").strip().lower() == split.lower()]
            else:
                print("  meta.csv columns not recognized; using all subject folders")
    if not subjects:
        subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dataset, "s0*"))
                          if os.path.isdir(p))
    return subjects[:limit] if limit else subjects


def ensure_predictions(subj):
    """Return the prediction dir for a subject, running TotalSegmentator if needed."""
    pred_dir = os.path.join(PRED_ROOT, subj)
    if glob.glob(os.path.join(pred_dir, "*.nii.gz")):
        return pred_dir
    if not RUN_TS:
        return None
    mri = os.path.join(DATASET, subj, "mri.nii.gz")
    if not os.path.exists(mri):
        return None
    os.makedirs(pred_dir, exist_ok=True)
    print(f"    running TotalSegmentator on {subj} ...")
    subprocess.run(["TotalSegmentator", "-i", mri, "-o", pred_dir] + TS_EXTRA, check=True)
    return pred_dir


def load_masks(mask_dir, ref_shape, min_voxels):
    """{organ: boolean mask} for every mask matching shape and at/above the size threshold."""
    out = {}
    for f in sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz"))):
        organ = os.path.basename(f).replace(".nii.gz", "")
        arr = nib.load(f).get_fdata() > 0.5
        if arr.shape == ref_shape and int(arr.sum()) >= min_voxels:
            out[organ] = arr
    return out


def analyze(pred, gt):
    """Per-organ Dice and status for every organ present in either mask set."""
    rows = []
    for organ in sorted(set(pred) | set(gt)):
        p, g = pred.get(organ), gt.get(organ)
        pv, gv = (int(p.sum()) if p is not None else 0), (int(g.sum()) if g is not None else 0)
        if pv == 0 and gv == 0:
            continue                                   # both absent -> not a real case
        if pv and gv:
            tp   = int(np.logical_and(p, g).sum())
            dice = 2 * tp / (pv + gv)
            iou  = tp / (pv + gv - tp)               # TP / (TP + FP + FN)
            status = "ok" if dice >= DICE_FLAG else "low_dice"
        elif pv == 0:
            dice, iou, status = 0.0, 0.0, "miss"      # gt has it, we missed it
        else:
            dice, iou, status = 0.0, 0.0, "false_positive"  # we invented it
        rows.append({"organ": organ, "dice": dice, "iou": iou,
                     "pred_vox": pv, "gt_vox": gv, "status": status})
    return rows


def best_slice(vol, axis):
    other = tuple(a for a in range(3) if a != axis)
    counts = np.count_nonzero(vol, axis=other)
    return int(np.argmax(counts)) if counts.max() > 0 else vol.shape[axis] // 2


def make_overlay(mri_img, pred, gt, subj, out_png):
    mri = mri_img.get_fdata()
    zooms = mri_img.header.get_zooms()[:3]
    names = sorted(set(pred) | set(gt))
    if not names:
        return
    base = plt.get_cmap("tab20")
    color_for = {n: base(i % 20) for i, n in enumerate(names)}
    gid = {n: i + 1 for i, n in enumerate(names)}

    def labelvol(masks):
        v = np.zeros(mri.shape, np.int32)
        for n, m in masks.items():
            v[m] = gid[n]
        return v

    pv, gv = labelvol(pred), labelvol(gt)
    combined = np.maximum(pv, gv)
    cmap = ListedColormap([(0, 0, 0, 0)] + [color_for[n] for n in names])
    norm = BoundaryNorm(np.arange(-0.5, len(names) + 1.5, 1), cmap.N)

    planes = [("axial", 2), ("coronal", 1), ("sagittal", 0)]
    titles = ["MRI", "Prediction", "Ground truth"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for r, (pn, axis) in enumerate(planes):
        idx = best_slice(combined, axis)
        dims = [d for d in range(3) if d != axis]
        aspect = zooms[dims[1]] / zooms[dims[0]]
        prep = lambda v: np.take(v, idx, axis=axis).T
        mslice = prep(mri)
        overlays = {"Prediction": prep(pv), "Ground truth": prep(gv)}
        for c, t in enumerate(titles):
            a = axes[r, c]
            a.imshow(mslice, cmap="gray", origin="lower", aspect=aspect)
            if t in overlays:
                o = overlays[t]
                a.imshow(np.ma.masked_where(o == 0, o), cmap=cmap, norm=norm,
                         alpha=0.55, origin="lower", aspect=aspect)
            if r == 0:
                a.set_title(t, fontsize=12)
            if c == 0:
                a.set_ylabel(f"{pn}\nslice {idx}", fontsize=10)
            a.set_xticks([]); a.set_yticks([])
    handles = [mpatches.Patch(color=color_for[n], label=n) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{subj} - prediction vs ground truth", fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def diagnose_pipeline(subjects):
    """Check affine alignment, organ-name overlap, spacing/shape, and task config.

    Called only when Task 2 detects a systematic shortfall (>30 % of matched organs
    flagged).  Inspects the first 3 subjects that have predictions.
    """
    print("\n=== PIPELINE DIAGNOSIS ===")
    checked = 0
    for subj in subjects:
        pred_dir = os.path.join(PRED_ROOT, subj)
        gt_dir   = os.path.join(DATASET, subj, "segmentations")
        mri_path = os.path.join(DATASET, subj, "mri.nii.gz")
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
        gt_files   = sorted(glob.glob(os.path.join(gt_dir,   "*.nii.gz")))
        if not (pred_files and gt_files and os.path.exists(mri_path)):
            continue
        checked += 1
        print(f"\n--- {subj} ---")

        # 1. Affine alignment
        pred_nib = nib.load(pred_files[0])
        mri_nib  = nib.load(mri_path)
        pred_organ = os.path.basename(pred_files[0]).replace(".nii.gz", "")
        gt_match = os.path.join(gt_dir, os.path.basename(pred_files[0]))
        pred_affine = pred_nib.affine
        mri_affine  = mri_nib.affine
        print(f"  Pred affine ({pred_organ}):\n{pred_affine}")
        print(f"  MRI  affine:\n{mri_affine}")
        print(f"  pred≈mri affine: {np.allclose(pred_affine, mri_affine, atol=1e-3)}")
        if os.path.exists(gt_match):
            gt_nib = nib.load(gt_match)
            gt_affine = gt_nib.affine
            print(f"  GT   affine ({pred_organ}):\n{gt_affine}")
            print(f"  pred≈gt  affine: {np.allclose(pred_affine, gt_affine, atol=1e-3)}")
            print(f"  gt≈mri   affine: {np.allclose(gt_affine,   mri_affine, atol=1e-3)}")

        # 2. Organ-name overlap
        pred_names = {os.path.basename(f).replace(".nii.gz", "") for f in pred_files}
        gt_names   = {os.path.basename(f).replace(".nii.gz", "") for f in gt_files}
        overlap    = pred_names & gt_names
        print(f"  Organ overlap: {len(overlap)} / {len(pred_names | gt_names)}  "
              f"(pred {len(pred_names)}, gt {len(gt_names)})")
        only_pred = pred_names - gt_names
        only_gt   = gt_names   - pred_names
        if only_pred:
            print(f"  Only in pred: {sorted(only_pred)[:5]} ...")
        if only_gt:
            print(f"  Only in gt:   {sorted(only_gt)[:5]} ...")

        # 3. Voxel spacing and shape
        print(f"  Pred shape {pred_nib.shape}  zooms {tuple(round(z,3) for z in pred_nib.header.get_zooms()[:3])}")
        print(f"  MRI  shape {mri_nib.shape}   zooms {tuple(round(z,3) for z in mri_nib.header.get_zooms()[:3])}")
        if os.path.exists(gt_match):
            print(f"  GT   shape {gt_nib.shape}    zooms {tuple(round(z,3) for z in gt_nib.header.get_zooms()[:3])}")

        if checked >= 3:
            break

    # 4. Task config check
    total_mr_organs = len(EXPECTED_DICE)
    avg_pred = sum(len(glob.glob(os.path.join(PRED_ROOT, s, "*.nii.gz")))
                   for s in subjects
                   if glob.glob(os.path.join(PRED_ROOT, s, "*.nii.gz"))) or 0
    n_with_pred = sum(1 for s in subjects
                      if glob.glob(os.path.join(PRED_ROOT, s, "*.nii.gz")))
    avg_pred = avg_pred / n_with_pred if n_with_pred else 0
    print(f"\n  TS_EXTRA = {TS_EXTRA}")
    print(f"  Expected ~{total_mr_organs} organs per subject for total_mr task")
    print(f"  Avg organ files found in predictions/: {avg_pred:.0f}")
    if avg_pred < total_mr_organs * 0.8:
        print("  WARNING: far fewer organ files than expected — "
              "predictions may be incomplete or use a combined-mask format.")


def write_expected_vs_measured(summary_path, report_dir):
    """Join measured summary_by_organ with EXPECTED_DICE; write comparison CSV.

    Returns the number of organs flagged (measured Dice > 0.15 below expected)
    among organs that appear in both tables, so main() can decide whether to run
    diagnose_pipeline().
    """
    out_path = os.path.join(report_dir, "expected_vs_measured.csv")
    rows = []
    with open(summary_path, newline="") as fh:
        for r in csv.DictReader(fh):
            organ = r["organ"]
            exp = EXPECTED_DICE.get(organ)
            n = r["n_present_both"]
            meas_str = r["mean_dice"]
            if not meas_str:
                meas = None
            else:
                meas = float(meas_str)
            delta = round(meas - exp, 4) if (meas is not None and exp is not None) else None
            rows.append({
                "organ": organ,
                "measured_dice": f"{meas:.4f}" if meas is not None else "",
                "expected_dice": f"{exp:.4f}" if exp is not None else "",
                "delta": f"{delta:.4f}" if delta is not None else "",
                "n_subjects": n,
            })

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["organ", "measured_dice", "expected_dice",
                                           "delta", "n_subjects"])
        w.writeheader()
        w.writerows(rows)

    # Report flags
    matched = [r for r in rows if r["measured_dice"] and r["expected_dice"]]
    flagged = [r for r in matched if float(r["delta"]) < -0.15]
    pct = 100 * len(flagged) / len(matched) if matched else 0
    systemic = pct > 30

    print(f"\n=== expected vs measured Dice (source: resources/results_all_classes_mr.json) ===")
    print(f"  Matched organs: {len(matched)}  |  Flagged (delta < -0.15): {len(flagged)}  "
          f"({pct:.0f}%)  =>  {'SYSTEMATIC shortfall' if systemic else 'isolated / acceptable'}")
    if flagged:
        print("  Flagged organs:")
        for r in sorted(flagged, key=lambda x: float(x["delta"])):
            print(f"    {r['organ']:32s}  measured {r['measured_dice']}  "
                  f"expected {r['expected_dice']}  delta {r['delta']}")
    print(f"  Written: {out_path}")
    return len(flagged), systemic


def main():
    os.makedirs(REPORT, exist_ok=True)
    overlay_dir = os.path.join(REPORT, "overlays")
    if SAVE_OVERLAYS:
        os.makedirs(overlay_dir, exist_ok=True)

    subjects = get_subjects(DATASET, SPLIT, LIMIT)
    print(f"Evaluating {len(subjects)} subject(s)\n")

    metrics_rows = []
    per_organ = defaultdict(lambda: {"dice": [], "iou": [], "ok": 0, "low_dice": 0,
                                     "miss": 0, "false_positive": 0})

    for i, subj in enumerate(subjects, 1):
        gt_dir = os.path.join(DATASET, subj, "segmentations")
        mri_path = os.path.join(DATASET, subj, "mri.nii.gz")
        if not (os.path.isdir(gt_dir) and os.path.exists(mri_path)):
            print(f"[{i}/{len(subjects)}] {subj}: missing mri or segmentations, skipping")
            continue
        pred_dir = ensure_predictions(subj)
        if pred_dir is None:
            print(f"[{i}/{len(subjects)}] {subj}: no predictions, skipping")
            continue

        mri_img = nib.load(mri_path)
        pred = load_masks(pred_dir, mri_img.shape, MIN_VOXELS)
        gt   = load_masks(gt_dir,   mri_img.shape, MIN_VOXELS)
        rows = analyze(pred, gt)

        for r in rows:
            metrics_rows.append({"subject": subj, **r})
            agg = per_organ[r["organ"]]
            agg[r["status"]] += 1
            if r["status"] in ("ok", "low_dice"):
                agg["dice"].append(r["dice"])
                agg["iou"].append(r["iou"])

        if SAVE_OVERLAYS:
            make_overlay(mri_img, pred, gt, subj, os.path.join(overlay_dir, f"{subj}.png"))

        present = [r["dice"] for r in rows if r["status"] in ("ok", "low_dice")]
        md = np.mean(present) if present else float("nan")
        flags = sum(1 for r in rows if r["status"] != "ok")
        print(f"[{i}/{len(subjects)}] {subj}: {len(rows)} organs | mean Dice {md:.3f} | {flags} flagged")

    # ---- IoU / Dice consistency check ----
    both_rows = [r for r in metrics_rows if r["status"] in ("ok", "low_dice")]
    bad = [r for r in both_rows
           if abs(r["iou"] - r["dice"] / (2 - r["dice"])) > 1e-6]
    if bad:
        print(f"FAIL: IoU/Dice mismatch in {len(bad)} rows:")
        for r in bad:
            print(r)
        raise SystemExit("Stopping — intersection/union counting is wrong.")
    if both_rows:
        dices = [r["dice"] for r in both_rows]
        ious  = [r["iou"]  for r in both_rows]
        rho, _ = spearmanr(dices, ious)
        print(f"\nIoU/Dice Spearman r = {rho:.6f}  (expect ~1.0)")

    # Round for CSV output (check above used precise float64 values)
    for r in metrics_rows:
        r["dice"] = round(r["dice"], 4)
        r["iou"]  = round(r["iou"],  4)

    # ---- per subject-organ metrics ----
    with open(os.path.join(REPORT, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "organ", "dice", "iou",
                                           "pred_vox", "gt_vox", "status"])
        w.writeheader()
        w.writerows(metrics_rows)

    # ---- per-organ rollup across subjects ----
    with open(os.path.join(REPORT, "summary_by_organ.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["organ", "n_present_both", "mean_dice", "mean_iou",
                    "n_ok", "n_low_dice", "n_miss", "n_false_positive"])
        for organ in sorted(per_organ):
            a = per_organ[organ]
            n = len(a["dice"])
            md   = sum(a["dice"]) / n if n else None
            miou = sum(a["iou"])  / n if n else None
            w.writerow([organ, n,
                        f"{md:.4f}"   if n else "",
                        f"{miou:.4f}" if n else "",
                        a["ok"], a["low_dice"], a["miss"], a["false_positive"]])

    # ---- expected vs measured comparison ----
    summary_path = os.path.join(REPORT, "summary_by_organ.csv")
    _, systemic = write_expected_vs_measured(summary_path, REPORT)

    # ---- console summary ----
    print("\n=== worst organs by mean Dice (present in both) ===")
    ranked = sorted(((organ, sum(a["dice"]) / len(a["dice"]))
                     for organ, a in per_organ.items() if a["dice"]),
                    key=lambda t: t[1])
    for organ, md in ranked[:10]:
        a = per_organ[organ]
        print(f"  {organ:28s} Dice {md:.3f}  "
              f"(miss {a['miss']}, false+ {a['false_positive']}, low {a['low_dice']})")
    total_flags = sum(a["miss"] + a["false_positive"] + a["low_dice"] for a in per_organ.values())
    print(f"\n{total_flags} total flagged organ-cases written to {REPORT}")
    print("  metrics.csv (per subject-organ) + summary_by_organ.csv + expected_vs_measured.csv + overlays/")

    if systemic:
        diagnose_pipeline(subjects)


if __name__ == "__main__":
    main()

    
