"""
TotalSegmentator CT evaluation harness.

Per subject:
  1. run TotalSegmentator if predictions are missing,
  2. score every structure against ground truth (Dice, IoU, TP/FP/FN, status),
  3. compute the ground-truth-free QC metrics from mask_metrics.py on the PREDICTED
     mask (these are the classifier's input features - they must be computable
     without ground truth, which is the whole point),
  4. save an overlay PNG for the first OVERLAY_LIMIT subjects.

Then:
  - combined_metrics.csv   one row per (subject, structure): labels + features
  - summary_by_organ.csv   per-organ rollup
  - lr_check.csv           left/right anatomical orientation audit
  - accept_analysis.csv    what the IoU >= IOU_ACCEPT filter selects

Everything is reoriented to canonical RAS before any voxel comparison, so Dice/IoU
and the positional features are computed in one consistent frame.

Run:  python evaluate_ct.py
"""

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import glob, os, csv, subprocess
from collections import defaultdict

from mask_metrics import (volume_metrics, shape_metrics, component_metrics,
                          boundary_metrics, intensity_metrics,
                          _EMPTY_INTENSITY_METRICS)

# ============================ CONFIG ============================
DATASET   = r"C:\Users\ansar\Downloads\Totalsegmentator_dataset_v201"
PRED_ROOT = r"C:\Users\ansar\Algoverse\predictions_ct"
REPORT    = r"C:\Users\ansar\Algoverse\report_ct"

IMAGE_NAME = "ct.nii.gz"            # CT dataset uses ct.nii.gz, not mri.nii.gz
TS_EXTRA   = ["-ta", "total"]       # CT task; add "-f" for the 3mm fast model

SPLIT   = "train"   # meta.csv split. Use train/val to BUILD the classifier,
                    # and keep "test" held out for the paper's evaluation.
LIMIT   = 20        # subjects for this pilot run; None = all
RUN_TS  = True

MIN_VOXELS    = 20     # below this a mask counts as absent (kills 1-voxel FOV slivers)
DICE_FLAG     = 0.50
IOU_ACCEPT    = 0.90   # label threshold: IoU >= this  ->  accept (1)
OVERLAY_LIMIT = 10     # how many subjects get an overlay PNG

CT_WINDOW = (-150, 250)   # HU window for display only (soft tissue)
# ================================================================

CANON = ("R", "A", "S")   # after reorientation: +x = patient Right, +y = Anterior, +z = Superior


# ---------------------------------------------------------------- subjects

def find_columns(header):
    lower = [h.lower() for h in header]
    id_col = next((header[lower.index(c)] for c in
                   ("image_id", "subject", "id", "image", "name") if c in lower), None)
    split_col = next((header[lower.index(c)] for c in
                      ("split", "partition", "subset", "set", "fold") if c in lower), None)
    return id_col, split_col


def get_subjects(dataset, split, limit):
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
    pred_dir = os.path.join(PRED_ROOT, subj)
    if glob.glob(os.path.join(pred_dir, "*.nii.gz")):
        return pred_dir
    if not RUN_TS:
        return None
    img = os.path.join(DATASET, subj, IMAGE_NAME)
    if not os.path.exists(img):
        return None
    os.makedirs(pred_dir, exist_ok=True)
    print(f"    running TotalSegmentator on {subj} ...")
    subprocess.run(["TotalSegmentator", "-i", img, "-o", pred_dir] + TS_EXTRA, check=True)
    return pred_dir


# ---------------------------------------------------------------- loading

def load_canonical(path):
    """Load a NIfTI reoriented to closest canonical RAS. Returns (img, original_axcodes)."""
    img = nib.load(path)
    return nib.as_closest_canonical(img), tuple(nib.aff2axcodes(img.affine))


def load_mask_dir(mask_dir, ref_shape, min_voxels):
    """{organ: bool array} in canonical RAS, thresholded by min_voxels."""
    out = {}
    for f in sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz"))):
        organ = os.path.basename(f)[:-len(".nii.gz")]
        img, _ = load_canonical(f)
        if img.shape != ref_shape:
            continue
        arr = np.asarray(img.dataobj) > 0.5
        if int(arr.sum()) >= min_voxels:
            out[organ] = arr
    return out


# ---------------------------------------------------------------- scoring

def score(p, g):
    """Dice, IoU, TP/FP/FN and status for one prediction/ground-truth pair."""
    pv = int(p.sum()) if p is not None else 0
    gv = int(g.sum()) if g is not None else 0
    if pv and gv:
        tp = int(np.logical_and(p, g).sum())
        fp, fn = pv - tp, gv - tp
        dice = 2 * tp / (pv + gv)
        iou = tp / (tp + fp + fn)
        status = "ok" if dice >= DICE_FLAG else "low_dice"
    elif pv == 0 and gv > 0:
        tp, fp, fn, dice, iou, status = 0, 0, gv, 0.0, 0.0, "miss"
    else:
        tp, fp, fn, dice, iou, status = 0, pv, 0, 0.0, 0.0, "false_positive"
    return {"dice": dice, "iou": iou, "tp": tp, "fp": fp, "fn": fn,
            "pred_vox": pv, "gt_vox": gv, "status": status}


def features(mask, spacing, ct):
    """Ground-truth-free QC features for one predicted mask (classifier inputs)."""
    f = {}
    f.update(volume_metrics(mask, spacing))
    f.update(shape_metrics(mask, spacing))
    f.update(component_metrics(mask))
    f.update(boundary_metrics(mask))
    f.update(intensity_metrics(mask, ct) if ct is not None else _EMPTY_INTENSITY_METRICS)
    return f


FEATURE_COLS = ["num_voxels", "volume_mm3", "is_empty",
                "centroid_x_rel", "centroid_y_rel", "centroid_z_rel",
                "bbox_x_rel", "bbox_y_rel", "bbox_z_rel",
                "bbox_volume_mm3", "mask_to_bbox_ratio",
                "num_components", "largest_component_fraction",
                "touches_boundary", "boundary_fraction",
                "mean_HU", "median_HU", "std_HU", "p05_HU", "p95_HU"]


# ---------------------------------------------------------------- overlay

def best_slice(vol, axis):
    other = tuple(a for a in range(3) if a != axis)
    counts = np.count_nonzero(vol, axis=other)
    return int(np.argmax(counts)) if counts.max() > 0 else vol.shape[axis] // 2


def make_overlay(ct, zooms, pred, gt, subj, out_png, max_organs=12):
    """Three-plane prediction vs ground-truth overlay. Arrays are canonical RAS."""
    names = sorted(set(pred) | set(gt))
    if not names:
        return
    # too many structures makes the figure unreadable - keep the largest ones
    if len(names) > max_organs:
        size = {n: int(pred.get(n, gt.get(n)).sum()) for n in names}
        names = sorted(sorted(names, key=lambda n: -size[n])[:max_organs])

    base = plt.get_cmap("tab20")
    color_for = {n: base(i % 20) for i, n in enumerate(names)}
    gid = {n: i + 1 for i, n in enumerate(names)}

    def labelvol(masks):
        v = np.zeros(ct.shape, np.int32)
        for n in names:
            if n in masks:
                v[masks[n]] = gid[n]
        return v

    pv, gv = labelvol(pred), labelvol(gt)
    both = np.maximum(pv, gv)
    cmap = ListedColormap([(0, 0, 0, 0)] + [color_for[n] for n in names])
    norm = BoundaryNorm(np.arange(-0.5, len(names) + 1.5, 1), cmap.N)
    ct_disp = np.clip(ct, *CT_WINDOW)          # CT needs windowing or it looks flat

    # canonical RAS: axis0 = L->R, axis1 = P->A, axis2 = I->S
    planes = [("axial (z)", 2), ("coronal (y)", 1), ("sagittal (x)", 0)]
    titles = ["CT", "Prediction", "Ground truth"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for r, (pn, axis) in enumerate(planes):
        idx = best_slice(both, axis)
        dims = [d for d in range(3) if d != axis]
        aspect = zooms[dims[1]] / zooms[dims[0]]
        prep = lambda v: np.take(v, idx, axis=axis).T
        base_sl = prep(ct_disp)
        ov = {"Prediction": prep(pv), "Ground truth": prep(gv)}
        for c, t in enumerate(titles):
            a = axes[r, c]
            a.imshow(base_sl, cmap="gray", origin="lower", aspect=aspect)
            if t in ov:
                o = ov[t]
                a.imshow(np.ma.masked_where(o == 0, o), cmap=cmap, norm=norm,
                         alpha=0.55, origin="lower", aspect=aspect)
            if r == 0:
                a.set_title(t, fontsize=12)
            if c == 0:
                a.set_ylabel(f"{pn}\nslice {idx}", fontsize=10)
            a.set_xticks([]); a.set_yticks([])
        # axial: mark which side of the image is the patient's right, so a
        # left/right label problem is visible instead of guessed at
        if axis == 2:
            for c in range(3):
                axes[r, c].text(0.02, 0.95, "pt LEFT", color="yellow", fontsize=8,
                                transform=axes[r, c].transAxes)
                axes[r, c].text(0.80, 0.95, "pt RIGHT", color="yellow", fontsize=8,
                                transform=axes[r, c].transAxes)

    handles = [mpatches.Patch(color=color_for[n], label=n) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{subj} - prediction vs ground truth (canonical RAS)", fontsize=14)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- left/right

def lr_rows(subj, source, masks):
    """
    Audit paired left/right structures in canonical RAS.

    In RAS the first axis increases toward the patient's RIGHT, so a correctly
    labelled *_right structure must have a HIGHER mean x index than its *_left
    partner. If left is higher, that pair's labels are anatomically swapped.
    """
    rows = []
    for name in sorted(masks):
        if not name.endswith("_left"):
            continue
        stem = name[:-len("_left")]
        rname = stem + "_right"
        if rname not in masks:
            continue
        xl = float(np.argwhere(masks[name])[:, 0].mean())
        xr = float(np.argwhere(masks[rname])[:, 0].mean())
        nx = masks[name].shape[0]
        rows.append({"subject": subj, "source": source, "structure": stem,
                     "left_centroid_x_rel": round(xl / nx, 4),
                     "right_centroid_x_rel": round(xr / nx, 4),
                     "right_minus_left": round((xr - xl) / nx, 4),
                     "verdict": "correct" if xr > xl else "SWAPPED"})
    return rows


# ---------------------------------------------------------------- main

def main():
    os.makedirs(REPORT, exist_ok=True)
    overlay_dir = os.path.join(REPORT, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    subjects = get_subjects(DATASET, SPLIT, LIMIT)
    print(f"Evaluating {len(subjects)} subject(s) from split={SPLIT}\n")

    rows, lr_all = [], []
    per_organ = defaultdict(lambda: {"dice": [], "iou": [], "ok": 0, "low_dice": 0,
                                     "miss": 0, "false_positive": 0, "accept": 0})

    for i, subj in enumerate(subjects, 1):
        gt_dir   = os.path.join(DATASET, subj, "segmentations")
        img_path = os.path.join(DATASET, subj, IMAGE_NAME)
        if not (os.path.isdir(gt_dir) and os.path.exists(img_path)):
            print(f"[{i}/{len(subjects)}] {subj}: missing {IMAGE_NAME} or segmentations, skipping")
            continue
        pred_dir = ensure_predictions(subj)
        if pred_dir is None:
            print(f"[{i}/{len(subjects)}] {subj}: no predictions, skipping")
            continue

        ct_img, orig_ax = load_canonical(img_path)
        ct = ct_img.get_fdata()
        spacing = ct_img.header.get_zooms()[:3]

        pred = load_mask_dir(pred_dir, ct_img.shape, MIN_VOXELS)
        gt   = load_mask_dir(gt_dir,   ct_img.shape, MIN_VOXELS)

        for organ in sorted(set(pred) | set(gt)):
            p, g = pred.get(organ), gt.get(organ)
            s = score(p, g)
            if s["pred_vox"] == 0 and s["gt_vox"] == 0:
                continue
            # features come from the PREDICTED mask only (no ground truth at inference).
            # a false-negative row has no predicted mask, so features are all-empty.
            f = features(p if p is not None else np.zeros(ct.shape, bool), spacing, ct)
            rows.append({"subject": subj, "organ": organ,
                         "accept": int(s["iou"] >= IOU_ACCEPT),
                         **s, **f, "orig_axcodes": "".join(orig_ax)})
            a = per_organ[organ]
            a[s["status"]] += 1
            if s["status"] in ("ok", "low_dice"):
                a["dice"].append(s["dice"]); a["iou"].append(s["iou"])
            if s["iou"] >= IOU_ACCEPT:
                a["accept"] += 1

        lr_all += lr_rows(subj, "pred", pred)
        lr_all += lr_rows(subj, "gt", gt)

        if i <= OVERLAY_LIMIT:
            make_overlay(ct, spacing, pred, gt, subj,
                         os.path.join(overlay_dir, f"{subj}.png"))

        pres = [r for r in rows if r["subject"] == subj and r["status"] in ("ok", "low_dice")]
        md = np.mean([r["dice"] for r in pres]) if pres else float("nan")
        na = sum(1 for r in rows if r["subject"] == subj and r["accept"])
        print(f"[{i}/{len(subjects)}] {subj}: {len(pres)} scored | mean Dice {md:.3f} | "
              f"{na} accepted (IoU>={IOU_ACCEPT}) | orient {''.join(orig_ax)}")

    if not rows:
        raise SystemExit("No rows produced - check paths.")

    # ---- IoU/Dice algebraic consistency (IoU = Dice / (2 - Dice)) ----
    both = [r for r in rows if r["status"] in ("ok", "low_dice")]
    bad = [r for r in both if abs(r["iou"] - r["dice"] / (2 - r["dice"])) > 1e-6]
    if bad:
        for r in bad[:10]:
            print("  MISMATCH", r["subject"], r["organ"], r["dice"], r["iou"])
        raise SystemExit(f"FAIL: IoU/Dice mismatch in {len(bad)} rows - counting is wrong.")
    print(f"\nIoU/Dice identity holds for all {len(both)} scored rows.")

    for r in rows:
        r["dice"] = round(r["dice"], 4); r["iou"] = round(r["iou"], 4)

    # ---- combined per subject-organ CSV (labels + features) ----
    cols = (["subject", "organ", "accept", "dice", "iou", "tp", "fp", "fn",
             "pred_vox", "gt_vox", "status"] + FEATURE_COLS + ["orig_axcodes"])
    with open(os.path.join(REPORT, "combined_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # ---- per-organ rollup ----
    with open(os.path.join(REPORT, "summary_by_organ.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["organ", "n_scored", "mean_dice", "mean_iou", "n_accept",
                    "accept_rate", "n_ok", "n_low_dice", "n_miss", "n_false_positive"])
        for organ in sorted(per_organ):
            a = per_organ[organ]
            n = len(a["dice"])
            total = a["ok"] + a["low_dice"] + a["miss"] + a["false_positive"]
            w.writerow([organ, n,
                        f"{sum(a['dice'])/n:.4f}" if n else "",
                        f"{sum(a['iou'])/n:.4f}" if n else "",
                        a["accept"], f"{a['accept']/total:.4f}" if total else "",
                        a["ok"], a["low_dice"], a["miss"], a["false_positive"]])

    # ---- left/right audit ----
    with open(os.path.join(REPORT, "lr_check.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "source", "structure",
                                           "left_centroid_x_rel", "right_centroid_x_rel",
                                           "right_minus_left", "verdict"])
        w.writeheader(); w.writerows(lr_all)

    swapped = [r for r in lr_all if r["verdict"] == "SWAPPED"]
    print("\n=== left/right audit (canonical RAS: +x = patient RIGHT) ===")
    print(f"  {len(lr_all)} paired structures checked, {len(swapped)} swapped "
          f"({100*len(swapped)/len(lr_all):.1f}%)" if lr_all else "  no L/R pairs found")
    for src in ("pred", "gt"):
        sub = [r for r in lr_all if r["source"] == src]
        sw = [r for r in sub if r["verdict"] == "SWAPPED"]
        if sub:
            print(f"  {src:4s}: {len(sw)}/{len(sub)} swapped")

    # ---- accept-filter analysis ----
    n_acc = sum(r["accept"] for r in rows)
    print(f"\n=== accept filter (IoU >= {IOU_ACCEPT}) ===")
    print(f"  {n_acc}/{len(rows)} masks accepted ({100*n_acc/len(rows):.1f}%)")
    print(f"  equivalent Dice threshold: {2*IOU_ACCEPT/(1+IOU_ACCEPT):.4f}")
    ious = sorted(r["iou"] for r in rows)
    qs = [np.percentile(ious, q) for q in (10, 25, 50, 75, 90)]
    print("  IoU percentiles p10/p25/p50/p75/p90: " + " ".join(f"{q:.3f}" for q in qs))

    with open(os.path.join(REPORT, "accept_analysis.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["organ", "n", "n_accept", "accept_rate", "mean_iou_accepted",
                    "mean_iou_rejected"])
        for organ in sorted(per_organ):
            o = [r for r in rows if r["organ"] == organ]
            acc = [r["iou"] for r in o if r["accept"]]
            rej = [r["iou"] for r in o if not r["accept"]]
            w.writerow([organ, len(o), len(acc), f"{len(acc)/len(o):.4f}",
                        f"{np.mean(acc):.4f}" if acc else "",
                        f"{np.mean(rej):.4f}" if rej else ""])

    print(f"\nWritten to {REPORT}:")
    print("  combined_metrics.csv  summary_by_organ.csv  lr_check.csv  "
          "accept_analysis.csv  overlays/")


if __name__ == "__main__":
    main()