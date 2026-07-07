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
            dice = 2 * int(np.logical_and(p, g).sum()) / (pv + gv)
            status = "ok" if dice >= DICE_FLAG else "low_dice"
        elif pv == 0:
            dice, status = 0.0, "miss"                 # gt has it, we missed it
        else:
            dice, status = 0.0, "false_positive"       # we invented it
        rows.append({"organ": organ, "dice": round(dice, 4),
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


def main():
    os.makedirs(REPORT, exist_ok=True)
    overlay_dir = os.path.join(REPORT, "overlays")
    if SAVE_OVERLAYS:
        os.makedirs(overlay_dir, exist_ok=True)

    subjects = get_subjects(DATASET, SPLIT, LIMIT)
    print(f"Evaluating {len(subjects)} subject(s)\n")

    metrics_rows = []
    per_organ = defaultdict(lambda: {"dice": [], "ok": 0, "low_dice": 0,
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

        if SAVE_OVERLAYS:
            make_overlay(mri_img, pred, gt, subj, os.path.join(overlay_dir, f"{subj}.png"))

        present = [r["dice"] for r in rows if r["status"] in ("ok", "low_dice")]
        md = np.mean(present) if present else float("nan")
        flags = sum(1 for r in rows if r["status"] != "ok")
        print(f"[{i}/{len(subjects)}] {subj}: {len(rows)} organs | mean Dice {md:.3f} | {flags} flagged")

    # ---- per subject-organ metrics ----
    with open(os.path.join(REPORT, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "organ", "dice",
                                           "pred_vox", "gt_vox", "status"])
        w.writeheader()
        w.writerows(metrics_rows)

    # ---- per-organ rollup across subjects ----
    with open(os.path.join(REPORT, "summary_by_organ.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["organ", "n_present_both", "mean_dice",
                    "n_ok", "n_low_dice", "n_miss", "n_false_positive"])
        for organ in sorted(per_organ):
            a = per_organ[organ]
            n = len(a["dice"])
            md = sum(a["dice"]) / n if n else None
            w.writerow([organ, n, f"{md:.4f}" if n else "",
                        a["ok"], a["low_dice"], a["miss"], a["false_positive"]])

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
    print("  metrics.csv (per subject-organ) + summary_by_organ.csv + overlays/")


if __name__ == "__main__":
    main()

    