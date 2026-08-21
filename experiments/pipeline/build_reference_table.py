"""
Stage 3 of the QC pipeline: build a per-organ reference table from GROUND-TRUTH masks.

Why: a feature like volume_mm3 = 50000 is meaningless on its own. It is normal for a
gallbladder and catastrophic for a liver. This script measures what each structure
actually looks like in ground truth, so a predicted mask can be scored as "3 SD smaller
than a typical pancreas" instead of "50000 mm3".

Run this against the dataset's TRAIN split, which stage 4 (curate_dataset.py) keeps
entirely separate from whatever split trains/tests the classifier. The table is a frozen
constant after that, exactly like the training-set mean/std in any normalisation step,
so using it at inference doesn't compromise the ground-truth-free claim: you look up the
table, you never need ground truth for the scan in front of you.

No GPU and no segmentation needed - only reads ground-truth masks that already exist, so
this stage has no dependency on stage 1/2 having run first.

Writes --output-csv (default: reference_stats.csv), long format, one row per
(organ, feature) pair. See totalsegmentator/qc_columns.py for the column glossary:
  Organ/Structure Name (organ), Feature/Metric Name (feature),
  Number of Reference Observations (n), Reference Median Value (median),
  Reference 25th/75th Percentile (q25/q75), Reference Interquartile Range (iqr),
  Computed in Log Space (log_space)

Also provides two importable helpers used by stage 4/5/6:
    load_reference(path)                 -> nested dict
    add_relative_features(df, ref)       -> df with *_z columns appended

Run:  python experiments/pipeline/build_reference_table.py --dataset-dir <dir> --modality ct --split train --output-csv reference/reference_stats.csv
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.common import add_dataset_args, resolve_modality, get_subjects, load_canonical, write_manifest
from totalsegmentator.mask_metrics import volume_metrics, shape_metrics, intensity_metrics
from totalsegmentator.qc_columns import (
    COL_ORGAN, COL_FEATURE_NAME, COL_OBSERVATION_COUNT, COL_REFERENCE_MEDIAN,
    COL_REFERENCE_Q25, COL_REFERENCE_Q75, COL_REFERENCE_IQR, COL_LOG_SPACE_FLAG,
    COL_NUM_VOXELS, COL_VOLUME_MM3,
    COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL,
    COL_BBOX_X_REL, COL_BBOX_Y_REL, COL_BBOX_Z_REL,
    COL_BBOX_VOLUME_MM3, COL_MASK_TO_BBOX_RATIO,
    COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU,
)

# Features that are roughly log-normal: take log1p before computing statistics, so the
# z-score is symmetric instead of being dragged around by the long right tail.
LOG_FEATURES = {COL_NUM_VOXELS, COL_VOLUME_MM3, COL_BBOX_VOLUME_MM3}

# Features worth normalising per organ. Deliberately excludes num_components and
# largest_component_fraction: on ground truth those are ~1 for every structure, so a
# reference is meaningless, and connected-component labelling is by far the slowest
# metric to compute. Those stay raw in the classifier, where they are already
# interpretable without organ context.
REFERENCE_FEATURES = [
    COL_NUM_VOXELS, COL_VOLUME_MM3,
    COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL,
    COL_BBOX_X_REL, COL_BBOX_Y_REL, COL_BBOX_Z_REL,
    COL_BBOX_VOLUME_MM3, COL_MASK_TO_BBOX_RATIO,
    COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU,
]


# ---------------------------------------------------------------- per subject

def measure_subject(args):
    """Return [(organ, {feature: value}), ...] for one subject's ground-truth masks.

    Module-level and picklable so it works with ProcessPoolExecutor on Windows.
    """
    dataset_dir, subj, image_name, min_voxels = args
    gt_dir = os.path.join(dataset_dir, subj, "segmentations")
    image_path = os.path.join(dataset_dir, subj, image_name)
    if not (os.path.isdir(gt_dir) and os.path.exists(image_path)):
        return subj, []

    try:
        image_img, _ = load_canonical(image_path)
        image_data = image_img.get_fdata()
        spacing = image_img.header.get_zooms()[:3]
    except Exception as e:
        return subj, [("__error__", {"msg": f"image load failed: {e}"})]

    out = []
    import glob
    for f in sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz"))):
        organ = os.path.basename(f)[:-len(".nii.gz")]
        try:
            m_img, _ = load_canonical(f)
            if m_img.shape != image_img.shape:
                continue
            mask = np.asarray(m_img.dataobj) > 0.5
        except Exception:
            continue
        if int(mask.sum()) < min_voxels:
            continue      # out of field of view, or a sliver: not a reference example
        rec = {}
        rec.update(volume_metrics(mask, spacing))
        rec.update(shape_metrics(mask, spacing))
        rec.update(intensity_metrics(mask, image_data))
        out.append((organ, {k: rec[k] for k in REFERENCE_FEATURES if k in rec}))
    return subj, out


# ---------------------------------------------------------------- reference table

def build_table(observations):
    """observations: {organ: {feature: [values]}} -> long-format rows."""
    rows = []
    for organ in sorted(observations):
        for feat in REFERENCE_FEATURES:
            vals = [v for v in observations[organ].get(feat, []) if v is not None]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            log_space = feat in LOG_FEATURES
            if log_space:
                arr = np.log1p(np.clip(arr, 0, None))
            q25, med, q75 = np.percentile(arr, [25, 50, 75])
            rows.append({COL_ORGAN: organ, COL_FEATURE_NAME: feat, COL_OBSERVATION_COUNT: len(arr),
                         COL_REFERENCE_MEDIAN: round(float(med), 6),
                         COL_REFERENCE_Q25: round(float(q25), 6),
                         COL_REFERENCE_Q75: round(float(q75), 6),
                         COL_REFERENCE_IQR: round(float(q75 - q25), 6),
                         COL_LOG_SPACE_FLAG: int(log_space)})
    return rows


# ---------------------------------------------------------------- apply (used by stage 4/5/6)

def load_reference(path):
    """reference_stats.csv -> {organ: {feature: {median, iqr, log_space}}}"""
    df = pd.read_csv(path)
    ref = {}
    # iterrows (not itertuples) because column names contain spaces/parentheses, which
    # itertuples cannot expose as attribute access.
    for _, row in df.iterrows():
        ref.setdefault(row[COL_ORGAN], {})[row[COL_FEATURE_NAME]] = {
            "median": row[COL_REFERENCE_MEDIAN], "iqr": row[COL_REFERENCE_IQR],
            "log_space": bool(row[COL_LOG_SPACE_FLAG])}
    return ref


def add_relative_features(df, ref, suffix="_z"):
    """Append organ-relative robust z-scores to a metrics dataframe.

    z = (x - organ_median) / (organ_IQR / 1.349)

    The 1.349 converts an IQR into a standard-deviation equivalent for a normal
    distribution, so the result reads on the familiar "how many SDs off" scale while
    staying robust to the outliers that raw std would chase.

    Raw columns are KEPT, not replaced. Organs absent from the reference table get NaN,
    which the imputer in the model pipeline (stage 5) already handles.
    """
    out = df.copy()
    for feat in REFERENCE_FEATURES:
        if feat not in out.columns:
            continue
        z = np.full(len(out), np.nan)
        for organ, stats in ref.items():
            s = stats.get(feat)
            if s is None or not np.isfinite(s["iqr"]) or s["iqr"] <= 0:
                continue      # constant feature for this organ: no meaningful scale
            sel = (out[COL_ORGAN] == organ).to_numpy()
            if not sel.any():
                continue
            x = pd.to_numeric(out.loc[sel, feat], errors="coerce").to_numpy(dtype=float)
            if s["log_space"]:
                x = np.log1p(np.clip(x, 0, None))
            z[sel] = (x - s["median"]) / (s["iqr"] / 1.349)
        out[feat + suffix] = z
    return out


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--output-csv", required=True, help="Path for reference_stats.csv.")
    parser.add_argument("--workers", type=int, default=10, help="Parallel subjects; 1 disables multiprocessing.")
    parser.add_argument("--min-voxels", type=int, default=20, help="Ignore FOV slivers when building the reference.")
    parser.add_argument("--min-n", type=int, default=10, help="Warn about organs with fewer than this many observations.")
    args = parser.parse_args()
    if args.split == "test":
        parser.error("Refusing to build the reference table from the test split - it must "
                     "stay held out. Use --split train (default) or another non-test split.")
    # --split all deliberately bypasses this check: it pools every subject regardless of
    # its original meta.csv split label (see get_subjects() in common.py), so there is no
    # "test split" concept to protect. By design the reference table is built from EVERY
    # subject (no --limit), including ones that also become classifier train/test
    # examples elsewhere in the pipeline - see RESEARCH.md for why this deliberate
    # overlap (and the small self-referential z-score bias it introduces, worst for
    # rare organs) was accepted in exchange for not wasting any subject's data.

    image_name, _ = resolve_modality(args)
    subjects = get_subjects(args.dataset_dir, args.split, args.limit, args.offset)
    print(f"Building reference from {len(subjects)} subject(s), split={args.split}")
    print("(ground-truth masks only, no segmentation, no GPU)\n")

    observations = {}
    def absorb(subj, recs):
        for organ, rec in recs:
            if organ == "__error__":
                print(f"  {subj}: {rec['msg']}")
                continue
            slot = observations.setdefault(organ, {})
            for k, v in rec.items():
                slot.setdefault(k, []).append(v)

    call_args = [(args.dataset_dir, s, image_name, args.min_voxels) for s in subjects]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(measure_subject, a): a[1] for a in call_args}
            for i, fut in enumerate(as_completed(futures), 1):
                subj, recs = fut.result()
                absorb(subj, recs)
                print(f"[{i}/{len(subjects)}] {subj}: {len(recs)} structures")
    else:
        for i, a in enumerate(call_args, 1):
            subj, recs = measure_subject(a)
            absorb(subj, recs)
            print(f"[{i}/{len(subjects)}] {subj}: {len(recs)} structures")

    rows = build_table(observations)
    if not rows:
        raise SystemExit("No observations collected - check --dataset-dir.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    counts = {o: max((len(v) for v in observations[o].values()), default=0) for o in observations}
    thin = sorted((n, o) for o, n in counts.items() if n < args.min_n)
    print(f"\n{len(counts)} structures, {len(rows)} organ-feature reference entries")
    print(f"median observations per structure: {int(np.median(list(counts.values())))}")
    if thin:
        print(f"\n{len(thin)} structures with fewer than {args.min_n} observations (their z-scores will be noisy):")
        for n, o in thin[:15]:
            print(f"  {o:35s} n={n}")

    write_manifest(
        os.path.join(os.path.dirname(os.path.abspath(args.output_csv)), "build_reference_manifest.json"),
        stage="build_reference_table", dataset_dir=args.dataset_dir, modality=args.modality,
        split=args.split, limit=args.limit, workers=args.workers, min_voxels=args.min_voxels,
        n_structures=len(counts), n_rows=len(rows),
    )
    print(f"\nwritten -> {args.output_csv}")


if __name__ == "__main__":
    main()
