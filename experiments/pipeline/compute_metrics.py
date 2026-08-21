"""
Stage 2 of the QC pipeline: score predictions against ground truth and compute the
ground-truth-free mask_metrics features on the PREDICTED mask - together, one row per
(subject, organ). This is the classifier's training data. Requires stage 1's
predictions to already exist on disk; never runs TotalSegmentator itself.

Everything is reoriented to canonical RAS (totalsegmentator.alignment style, matching
mask_metrics.py) before any voxel comparison, so Dice/IoU and the positional features
are computed in one consistent frame regardless of how a scan happens to be stored.

Writes (see totalsegmentator/qc_columns.py for the full column-name glossary):

  <output-csv> (default: combined_metrics.csv) - one row per (subject, organ):
    Subject ID, Organ/Structure Name, Accept/Reject Label, Sorensen-Dice Coefficient,
    Intersection over Union, True/False Positive/Negative Voxel Counts,
    Predicted/Ground Truth Mask Voxel Count, Match Status,
    every mask_metrics.py feature computed on the PREDICTED mask,
    Original Image Axis Orientation Codes

  --summary-csv (default: summary_by_organ.csv) - one row per organ, aggregated:
    Organ/Structure Name, Number of Masks Scored, Mean Dice/IoU, Number Accepted,
    Accept Rate, status breakdown (n_ok/n_low_dice/n_miss/n_false_positive)

  --overlays-dir (optional) - one <subject>.png per subject (capped at --overlay-limit):
    three-plane prediction-vs-ground-truth overlay, canonical RAS, "pt LEFT"/"pt RIGHT"
    markers on the axial view.

  --lr-audit-csv (optional) - one row per paired left/right structure per subject per
    source (prediction or ground truth): Subject ID, Mask Source, Paired Structure Base
    Name, Left/Right Relative Centroid X, Right Minus Left Difference, Verdict.

  --accept-analysis-csv (optional) - one row per organ: what the --iou-accept filter
    selects (Number of Masks, Number Accepted, Accept Rate, Mean IoU of
    Accepted/Rejected Masks).

  --expected-dice-json (optional, e.g. resources/expected_dice_mr.json) - if given,
  also writes --expected-vs-measured-csv comparing measured vs. published per-organ
  Dice, and runs an automatic pipeline diagnosis (affine alignment, organ-name overlap,
  shape/spacing) if >30% of matched organs are flagged as a systematic shortfall.

Run:  python experiments/pipeline/compute_metrics.py --dataset-dir <dir> --predictions-dir <dir> --modality mr --split test --output-csv metrics/combined_metrics.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.common import add_dataset_args, resolve_modality, get_subjects, load_canonical, write_manifest
from totalsegmentator.mask_metrics import (volume_metrics, shape_metrics, component_metrics,
                          boundary_metrics, intensity_metrics, _EMPTY_INTENSITY_METRICS)
from totalsegmentator.qc_columns import (
    COL_SUBJECT, COL_ORGAN, COL_ORIG_AXCODES,
    COL_DICE, COL_IOU, COL_TRUE_POSITIVES, COL_FALSE_POSITIVES, COL_FALSE_NEGATIVES,
    COL_PREDICTED_VOXEL_COUNT, COL_GROUND_TRUTH_VOXEL_COUNT, COL_MATCH_STATUS,
    COL_ACCEPT_LABEL,
    COL_NUM_VOXELS, COL_VOLUME_MM3, COL_IS_EMPTY,
    COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL,
    COL_BBOX_X_REL, COL_BBOX_Y_REL, COL_BBOX_Z_REL,
    COL_BBOX_VOLUME_MM3, COL_MASK_TO_BBOX_RATIO,
    COL_NUM_COMPONENTS, COL_LARGEST_COMPONENT_FRACTION,
    COL_TOUCHES_BOUNDARY, COL_BOUNDARY_FRACTION,
    COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU,
    COL_MASK_SOURCE, COL_PAIRED_STRUCTURE,
    COL_LEFT_CENTROID_X_REL, COL_RIGHT_CENTROID_X_REL, COL_RIGHT_MINUS_LEFT, COL_LR_VERDICT,
    COL_N_SCORED, COL_MEAN_DICE, COL_MEAN_IOU,
    COL_N_OK, COL_N_LOW_DICE, COL_N_MISS, COL_N_FALSE_POSITIVE,
    COL_N_ACCEPT, COL_ACCEPT_RATE,
    COL_N_MASKS, COL_MEAN_IOU_ACCEPTED, COL_MEAN_IOU_REJECTED,
    COL_MEASURED_DICE, COL_EXPECTED_DICE, COL_DICE_DELTA, COL_N_SUBJECTS,
)

FEATURE_COLS = [COL_NUM_VOXELS, COL_VOLUME_MM3, COL_IS_EMPTY,
                COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL,
                COL_BBOX_X_REL, COL_BBOX_Y_REL, COL_BBOX_Z_REL,
                COL_BBOX_VOLUME_MM3, COL_MASK_TO_BBOX_RATIO,
                COL_NUM_COMPONENTS, COL_LARGEST_COMPONENT_FRACTION,
                COL_TOUCHES_BOUNDARY, COL_BOUNDARY_FRACTION,
                COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU]


# ---------------------------------------------------------------- loading

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

def score(pred_mask, gt_mask, dice_flag):
    """Dice, IoU, TP/FP/FN and status for one prediction/ground-truth pair."""
    predicted_voxel_count = int(pred_mask.sum()) if pred_mask is not None else 0
    ground_truth_voxel_count = int(gt_mask.sum()) if gt_mask is not None else 0
    if predicted_voxel_count and ground_truth_voxel_count:
        true_positives = int(np.logical_and(pred_mask, gt_mask).sum())
        false_positives = predicted_voxel_count - true_positives
        false_negatives = ground_truth_voxel_count - true_positives
        dice = 2 * true_positives / (predicted_voxel_count + ground_truth_voxel_count)
        iou = true_positives / (true_positives + false_positives + false_negatives)
        status = "ok" if dice >= dice_flag else "low_dice"
    elif predicted_voxel_count == 0 and ground_truth_voxel_count > 0:
        true_positives, false_positives, false_negatives = 0, 0, ground_truth_voxel_count
        dice, iou, status = 0.0, 0.0, "miss"
    else:
        true_positives, false_positives, false_negatives = 0, predicted_voxel_count, 0
        dice, iou, status = 0.0, 0.0, "false_positive"
    return {COL_DICE: dice, COL_IOU: iou,
            COL_TRUE_POSITIVES: true_positives, COL_FALSE_POSITIVES: false_positives,
            COL_FALSE_NEGATIVES: false_negatives,
            COL_PREDICTED_VOXEL_COUNT: predicted_voxel_count,
            COL_GROUND_TRUTH_VOXEL_COUNT: ground_truth_voxel_count,
            COL_MATCH_STATUS: status}


def features(mask, spacing, image):
    """Ground-truth-free QC features for one predicted mask (classifier inputs)."""
    f = {}
    f.update(volume_metrics(mask, spacing))
    f.update(shape_metrics(mask, spacing))
    f.update(component_metrics(mask))
    f.update(boundary_metrics(mask))
    f.update(intensity_metrics(mask, image) if image is not None else _EMPTY_INTENSITY_METRICS)
    return f


# ---------------------------------------------------------------- per-subject worker
# Module-level and picklable so it works with ProcessPoolExecutor (same pattern as
# build_reference_table.py::measure_subject). Bundles everything one subject needs to
# be scored independently of every other subject, so --workers > 1 can run subjects
# across separate processes/cores instead of this stage's single-threaded default.

def score_subject(call_args):
    (dataset_dir, predictions_dir, subj, image_name, min_voxels, dice_flag, iou_accept,
     want_lr_audit, want_overlay, overlays_dir, hu_window) = call_args

    gt_dir = os.path.join(dataset_dir, subj, "segmentations")
    image_path = os.path.join(dataset_dir, subj, image_name)
    pred_dir = os.path.join(predictions_dir, subj)
    if not (os.path.isdir(gt_dir) and os.path.exists(image_path)):
        return subj, [], [], f"missing {image_name} or segmentations, skipping"
    if not glob.glob(os.path.join(pred_dir, "*.nii.gz")):
        return subj, [], [], f"no predictions in {pred_dir}, skipping (run stage 1 first)"

    image_img, orig_ax = load_canonical(image_path)
    image_data = image_img.get_fdata()
    spacing = image_img.header.get_zooms()[:3]

    pred = load_mask_dir(pred_dir, image_img.shape, min_voxels)
    gt = load_mask_dir(gt_dir, image_img.shape, min_voxels)

    subj_rows = []
    for organ in sorted(set(pred) | set(gt)):
        pred_mask, gt_mask = pred.get(organ), gt.get(organ)
        s = score(pred_mask, gt_mask, dice_flag)
        if s[COL_PREDICTED_VOXEL_COUNT] == 0 and s[COL_GROUND_TRUTH_VOXEL_COUNT] == 0:
            continue
        f = features(pred_mask if pred_mask is not None else np.zeros(image_img.shape, bool), spacing, image_data)
        subj_rows.append({COL_SUBJECT: subj, COL_ORGAN: organ,
                          COL_ACCEPT_LABEL: int(s[COL_IOU] >= iou_accept),
                          **s, **f, COL_ORIG_AXCODES: "".join(orig_ax)})

    subj_lr_rows = []
    if want_lr_audit:
        subj_lr_rows += lr_rows(subj, "pred", pred)
        subj_lr_rows += lr_rows(subj, "gt", gt)

    if want_overlay:
        make_overlay(image_data, spacing, pred, gt, subj,
                     os.path.join(overlays_dir, f"{subj}.png"), hu_window=hu_window)

    return subj, subj_rows, subj_lr_rows, None


# ---------------------------------------------------------------- overlay

def best_slice(vol, axis):
    other = tuple(a for a in range(3) if a != axis)
    counts = np.count_nonzero(vol, axis=other)
    return int(np.argmax(counts)) if counts.max() > 0 else vol.shape[axis] // 2


def make_overlay(image, zooms, pred, gt, subj, out_png, hu_window=None, max_organs=12):
    """Three-plane prediction vs ground-truth overlay. Arrays are canonical RAS."""
    names = sorted(set(pred) | set(gt))
    if not names:
        return
    if len(names) > max_organs:
        size = {n: int(pred.get(n, gt.get(n)).sum()) for n in names}
        names = sorted(sorted(names, key=lambda n: -size[n])[:max_organs])

    base = plt.get_cmap("tab20")
    color_for = {n: base(i % 20) for i, n in enumerate(names)}
    gid = {n: i + 1 for i, n in enumerate(names)}

    def labelvol(masks):
        v = np.zeros(image.shape, np.int32)
        for n in names:
            if n in masks:
                v[masks[n]] = gid[n]
        return v

    pred_label_vol, gt_label_vol = labelvol(pred), labelvol(gt)
    both = np.maximum(pred_label_vol, gt_label_vol)
    cmap = ListedColormap([(0, 0, 0, 0)] + [color_for[n] for n in names])
    norm = BoundaryNorm(np.arange(-0.5, len(names) + 1.5, 1), cmap.N)
    disp = np.clip(image, *hu_window) if hu_window else image

    planes = [("axial (z)", 2), ("coronal (y)", 1), ("sagittal (x)", 0)]
    titles = ["Image", "Prediction", "Ground truth"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for r, (pn, axis) in enumerate(planes):
        idx = best_slice(both, axis)
        dims = [d for d in range(3) if d != axis]
        aspect = zooms[dims[1]] / zooms[dims[0]]
        prep = lambda v: np.take(v, idx, axis=axis).T
        base_sl = prep(disp)
        ov = {"Prediction": prep(pred_label_vol), "Ground truth": prep(gt_label_vol)}
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
    """Audit paired left/right structures in canonical RAS (+x = patient RIGHT)."""
    rows = []
    for name in sorted(masks):
        if not name.endswith("_left"):
            continue
        stem = name[:-len("_left")]
        rname = stem + "_right"
        if rname not in masks:
            continue
        left_centroid_x = float(np.argwhere(masks[name])[:, 0].mean())
        right_centroid_x = float(np.argwhere(masks[rname])[:, 0].mean())
        axis_length = masks[name].shape[0]
        rows.append({COL_SUBJECT: subj, COL_MASK_SOURCE: source, COL_PAIRED_STRUCTURE: stem,
                     COL_LEFT_CENTROID_X_REL: round(left_centroid_x / axis_length, 4),
                     COL_RIGHT_CENTROID_X_REL: round(right_centroid_x / axis_length, 4),
                     COL_RIGHT_MINUS_LEFT: round((right_centroid_x - left_centroid_x) / axis_length, 4),
                     COL_LR_VERDICT: "correct" if right_centroid_x > left_centroid_x else "SWAPPED"})
    return rows


# ---------------------------------------------------------------- pipeline diagnosis

def diagnose_pipeline(subjects, dataset_dir, predictions_dir, image_name, expected_organs):
    """Check affine alignment, organ-name overlap, spacing/shape for the first 3 subjects
    that have predictions. Only called when the expected-vs-measured comparison finds a
    systematic (>30% of organs) shortfall against the benchmark.
    """
    print("\n=== PIPELINE DIAGNOSIS ===")
    checked = 0
    for subj in subjects:
        pred_dir = os.path.join(predictions_dir, subj)
        gt_dir = os.path.join(dataset_dir, subj, "segmentations")
        image_path = os.path.join(dataset_dir, subj, image_name)
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz")))
        if not (pred_files and gt_files and os.path.exists(image_path)):
            continue
        checked += 1
        print(f"\n--- {subj} ---")

        pred_nib = nib.load(pred_files[0])
        image_nib = nib.load(image_path)
        pred_organ = os.path.basename(pred_files[0]).replace(".nii.gz", "")
        gt_match = os.path.join(gt_dir, os.path.basename(pred_files[0]))
        print(f"  Pred affine ({pred_organ}):\n{pred_nib.affine}")
        print(f"  Image affine:\n{image_nib.affine}")
        print(f"  pred≈image affine: {np.allclose(pred_nib.affine, image_nib.affine, atol=1e-3)}")
        if os.path.exists(gt_match):
            gt_nib = nib.load(gt_match)
            print(f"  GT   affine ({pred_organ}):\n{gt_nib.affine}")
            print(f"  pred≈gt    affine: {np.allclose(pred_nib.affine, gt_nib.affine, atol=1e-3)}")
            print(f"  gt≈image   affine: {np.allclose(gt_nib.affine, image_nib.affine, atol=1e-3)}")

        pred_names = {os.path.basename(f).replace(".nii.gz", "") for f in pred_files}
        gt_names = {os.path.basename(f).replace(".nii.gz", "") for f in gt_files}
        overlap = pred_names & gt_names
        print(f"  Organ overlap: {len(overlap)} / {len(pred_names | gt_names)}  "
              f"(pred {len(pred_names)}, gt {len(gt_names)})")

        print(f"  Pred shape {pred_nib.shape}  zooms {tuple(round(z,3) for z in pred_nib.header.get_zooms()[:3])}")
        print(f"  Image shape {image_nib.shape}   zooms {tuple(round(z,3) for z in image_nib.header.get_zooms()[:3])}")
        if os.path.exists(gt_match):
            print(f"  GT   shape {gt_nib.shape}    zooms {tuple(round(z,3) for z in gt_nib.header.get_zooms()[:3])}")
        if checked >= 3:
            break

    avg_pred = sum(len(glob.glob(os.path.join(predictions_dir, s, "*.nii.gz"))) for s in subjects) or 0
    n_with_pred = sum(1 for s in subjects if glob.glob(os.path.join(predictions_dir, s, "*.nii.gz")))
    avg_pred = avg_pred / n_with_pred if n_with_pred else 0
    print(f"\n  Expected ~{expected_organs} organs per subject")
    print(f"  Avg organ files found in predictions: {avg_pred:.0f}")
    if avg_pred < expected_organs * 0.8:
        print("  WARNING: far fewer organ files than expected — "
              "predictions may be incomplete or use a combined-mask format.")


def write_expected_vs_measured(summary_rows, expected_dice, out_path):
    """Join measured per-organ Dice with a published benchmark; write comparison CSV.

    Returns (n_flagged, systemic) so main() can decide whether to run diagnose_pipeline.
    """
    rows = []
    for r in summary_rows:
        organ = r[COL_ORGAN]
        expected = expected_dice.get(organ)
        measured_str = r[COL_MEAN_DICE]
        measured = float(measured_str) if measured_str else None
        delta = round(measured - expected, 4) if (measured is not None and expected is not None) else None
        rows.append({
            COL_ORGAN: organ,
            COL_MEASURED_DICE: f"{measured:.4f}" if measured is not None else "",
            COL_EXPECTED_DICE: f"{expected:.4f}" if expected is not None else "",
            COL_DICE_DELTA: f"{delta:.4f}" if delta is not None else "",
            COL_N_SUBJECTS: r[COL_N_SCORED],
        })
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[COL_ORGAN, COL_MEASURED_DICE, COL_EXPECTED_DICE,
                                           COL_DICE_DELTA, COL_N_SUBJECTS])
        w.writeheader(); w.writerows(rows)

    matched = [r for r in rows if r[COL_MEASURED_DICE] and r[COL_EXPECTED_DICE]]
    flagged = [r for r in matched if float(r[COL_DICE_DELTA]) < -0.15]
    pct = 100 * len(flagged) / len(matched) if matched else 0
    systemic = pct > 30
    print(f"\n=== expected vs measured Dice ===")
    print(f"  Matched organs: {len(matched)}  |  Flagged (delta < -0.15): {len(flagged)}  "
          f"({pct:.0f}%)  =>  {'SYSTEMATIC shortfall' if systemic else 'isolated / acceptable'}")
    if flagged:
        for r in sorted(flagged, key=lambda x: float(x[COL_DICE_DELTA])):
            print(f"    {r[COL_ORGAN]:32s}  measured {r[COL_MEASURED_DICE]}  "
                  f"expected {r[COL_EXPECTED_DICE]}  delta {r[COL_DICE_DELTA]}")
    print(f"  Written: {out_path}")
    return len(flagged), systemic


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--predictions-dir", required=True, help="Output of stage 1 (run_inference.py).")
    parser.add_argument("--output-csv", required=True, help="Path for combined_metrics.csv.")
    parser.add_argument("--summary-csv", default=None,
                        help="Path for the per-organ rollup. Default: summary_by_organ.csv next to --output-csv.")
    parser.add_argument("--min-voxels", type=int, default=20,
                        help="Masks below this voxel count are treated as absent (kills edge slivers).")
    parser.add_argument("--dice-flag", type=float, default=0.50, help="Dice below this is flagged low_dice.")
    parser.add_argument("--iou-accept", type=float, default=0.90, help="IoU threshold for the accept/reject label.")
    parser.add_argument("--overlays-dir", default=None, help="If given, save prediction-vs-GT overlay PNGs here.")
    parser.add_argument("--overlay-limit", type=int, default=10, help="Max number of subjects to render overlays for.")
    parser.add_argument("--hu-window", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="HU window for overlay display only (e.g. --hu-window -150 250 for CT soft tissue).")
    parser.add_argument("--lr-audit-csv", default=None, help="If given, write the left/right orientation audit here.")
    parser.add_argument("--accept-analysis-csv", default=None, help="If given, write the accept-filter breakdown here.")
    parser.add_argument("--expected-dice-json", default=None,
                        help="If given (e.g. resources/expected_dice_mr.json), compare measured Dice against it.")
    parser.add_argument("--expected-vs-measured-csv", default=None,
                        help="Path for the expected-vs-measured comparison. Default: expected_vs_measured.csv next to --output-csv.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel subjects via ProcessPoolExecutor; 1 (default) disables multiprocessing. "
                             "Note: with --workers > 1, --overlay-limit selects the first N subjects to FINISH, "
                             "not the first N in --split order (subjects complete out of order across processes).")
    args = parser.parse_args()

    image_name, task = resolve_modality(args)
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = args.summary_csv or os.path.join(out_dir, "summary_by_organ.csv")
    if args.overlays_dir:
        os.makedirs(args.overlays_dir, exist_ok=True)

    subjects = get_subjects(args.dataset_dir, args.split, args.limit, args.offset)
    print(f"Computing metrics for {len(subjects)} subject(s) | modality={args.modality} split={args.split}\n")

    rows, lr_all = [], []
    per_organ = defaultdict(lambda: {"dice": [], "iou": [], "ok": 0, "low_dice": 0,
                                     "miss": 0, "false_positive": 0, "accept": 0})

    def absorb(subj, subj_rows, subj_lr_rows, err, i):
        if err:
            print(f"[{i}/{len(subjects)}] {subj}: {err}")
            return
        rows.extend(subj_rows)
        for r in subj_rows:
            a = per_organ[r[COL_ORGAN]]
            a[r[COL_MATCH_STATUS]] += 1
            if r[COL_MATCH_STATUS] in ("ok", "low_dice"):
                a["dice"].append(r[COL_DICE]); a["iou"].append(r[COL_IOU])
            if r[COL_ACCEPT_LABEL]:
                a["accept"] += 1
        if args.lr_audit_csv:
            lr_all.extend(subj_lr_rows)
        present = [r for r in subj_rows if r[COL_MATCH_STATUS] in ("ok", "low_dice")]
        mean_dice = np.mean([r[COL_DICE] for r in present]) if present else float("nan")
        n_accepted = sum(1 for r in subj_rows if r[COL_ACCEPT_LABEL])
        orient = subj_rows[0][COL_ORIG_AXCODES] if subj_rows else "?"
        print(f"[{i}/{len(subjects)}] {subj}: {len(present)} scored | mean Dice {mean_dice:.3f} | "
              f"{n_accepted} accepted (IoU>={args.iou_accept}) | orient {orient}")

    call_args = [(args.dataset_dir, args.predictions_dir, s, image_name, args.min_voxels,
                  args.dice_flag, args.iou_accept, bool(args.lr_audit_csv),
                  bool(args.overlays_dir) and idx <= args.overlay_limit,
                  args.overlays_dir, args.hu_window)
                 for idx, s in enumerate(subjects, 1)]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(score_subject, a): a[2] for a in call_args}
            for i, fut in enumerate(as_completed(futures), 1):
                subj, subj_rows, subj_lr_rows, err = fut.result()
                absorb(subj, subj_rows, subj_lr_rows, err, i)
    else:
        for i, a in enumerate(call_args, 1):
            subj, subj_rows, subj_lr_rows, err = score_subject(a)
            absorb(subj, subj_rows, subj_lr_rows, err, i)

    if not rows:
        raise SystemExit("No rows produced - check --dataset-dir/--predictions-dir.")

    # IoU/Dice algebraic consistency (IoU = Dice / (2 - Dice)) - guards against a broken
    # TP/FP/FN count silently producing wrong numbers.
    both = [r for r in rows if r[COL_MATCH_STATUS] in ("ok", "low_dice")]
    bad = [r for r in both if abs(r[COL_IOU] - r[COL_DICE] / (2 - r[COL_DICE])) > 1e-6]
    if bad:
        for r in bad[:10]:
            print("  MISMATCH", r[COL_SUBJECT], r[COL_ORGAN], r[COL_DICE], r[COL_IOU])
        raise SystemExit(f"FAIL: IoU/Dice mismatch in {len(bad)} rows - counting is wrong.")
    print(f"\nIoU/Dice identity holds for all {len(both)} scored rows.")

    for r in rows:
        r[COL_DICE] = round(r[COL_DICE], 4); r[COL_IOU] = round(r[COL_IOU], 4)

    cols = ([COL_SUBJECT, COL_ORGAN, COL_ACCEPT_LABEL, COL_DICE, COL_IOU,
             COL_TRUE_POSITIVES, COL_FALSE_POSITIVES, COL_FALSE_NEGATIVES,
             COL_PREDICTED_VOXEL_COUNT, COL_GROUND_TRUTH_VOXEL_COUNT, COL_MATCH_STATUS]
            + FEATURE_COLS + [COL_ORIG_AXCODES])
    with open(args.output_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    summary_rows = []
    with open(summary_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([COL_ORGAN, COL_N_SCORED, COL_MEAN_DICE, COL_MEAN_IOU, COL_N_ACCEPT,
                    COL_ACCEPT_RATE, COL_N_OK, COL_N_LOW_DICE, COL_N_MISS, COL_N_FALSE_POSITIVE])
        for organ in sorted(per_organ):
            a = per_organ[organ]
            n = len(a["dice"])
            total = a["ok"] + a["low_dice"] + a["miss"] + a["false_positive"]
            mean_dice = f"{sum(a['dice'])/n:.4f}" if n else ""
            mean_iou = f"{sum(a['iou'])/n:.4f}" if n else ""
            accept_rate = f"{a['accept']/total:.4f}" if total else ""
            w.writerow([organ, n, mean_dice, mean_iou, a["accept"], accept_rate,
                        a["ok"], a["low_dice"], a["miss"], a["false_positive"]])
            summary_rows.append({COL_ORGAN: organ, COL_N_SCORED: n, COL_MEAN_DICE: mean_dice,
                                 COL_MEAN_IOU: mean_iou})

    if args.lr_audit_csv:
        with open(args.lr_audit_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[COL_SUBJECT, COL_MASK_SOURCE, COL_PAIRED_STRUCTURE,
                                               COL_LEFT_CENTROID_X_REL, COL_RIGHT_CENTROID_X_REL,
                                               COL_RIGHT_MINUS_LEFT, COL_LR_VERDICT])
            w.writeheader(); w.writerows(lr_all)
        swapped = [r for r in lr_all if r[COL_LR_VERDICT] == "SWAPPED"]
        print(f"\n=== left/right audit ===")
        print(f"  {len(lr_all)} paired structures checked, {len(swapped)} swapped"
              if lr_all else "  no L/R pairs found")

    if args.accept_analysis_csv:
        with open(args.accept_analysis_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([COL_ORGAN, COL_N_MASKS, COL_N_ACCEPT, COL_ACCEPT_RATE,
                        COL_MEAN_IOU_ACCEPTED, COL_MEAN_IOU_REJECTED])
            for organ in sorted(per_organ):
                o = [r for r in rows if r[COL_ORGAN] == organ]
                acc = [r[COL_IOU] for r in o if r[COL_ACCEPT_LABEL]]
                rej = [r[COL_IOU] for r in o if not r[COL_ACCEPT_LABEL]]
                w.writerow([organ, len(o), len(acc), f"{len(acc)/len(o):.4f}",
                            f"{np.mean(acc):.4f}" if acc else "",
                            f"{np.mean(rej):.4f}" if rej else ""])

    systemic = False
    if args.expected_dice_json:
        with open(args.expected_dice_json) as fh:
            expected_dice = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
        expected_vs_measured_csv = args.expected_vs_measured_csv or os.path.join(out_dir, "expected_vs_measured.csv")
        _, systemic = write_expected_vs_measured(summary_rows, expected_dice, expected_vs_measured_csv)
        if systemic:
            diagnose_pipeline(subjects, args.dataset_dir, args.predictions_dir, image_name, len(expected_dice))

    write_manifest(
        os.path.join(out_dir, "compute_metrics_manifest.json"),
        stage="compute_metrics", dataset_dir=args.dataset_dir, predictions_dir=args.predictions_dir,
        modality=args.modality, image_name=image_name, split=args.split, limit=args.limit,
        min_voxels=args.min_voxels, dice_flag=args.dice_flag, iou_accept=args.iou_accept,
        n_subjects=len(subjects), n_rows=len(rows), systemic_shortfall=systemic,
    )
    print(f"\n{len(rows)} rows written to {args.output_csv}")


if __name__ == "__main__":
    main()
