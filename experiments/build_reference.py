"""
Build a per-organ reference table from GROUND-TRUTH masks.

Why: a feature like volume_mm3 = 50000 is meaningless on its own. It is normal for a
gallbladder and catastrophic for a liver. This script measures what each structure
actually looks like in ground truth, so a predicted mask can be scored as "3 SD smaller
than a typical pancreas" instead of "50000 mm3".

The table is built from the TotalSegmentator TRAIN split, which is held out from the
classifier's train/val/test data entirely. It is a frozen constant after that, exactly
like the training-set mean and standard deviation in any normalisation step, so it does
not compromise the ground-truth-free claim at inference: at deployment you look up the
table, you never need ground truth for the scan in front of you.

No GPU and no segmentation needed. It only reads ground-truth masks that already exist.

Outputs reference_stats.csv (long format):
    organ, feature, n, median, q25, q75, iqr, log_space

Also provides two helpers that train_classifier.py can import:
    load_reference(path)                 -> nested dict
    add_relative_features(df, ref)       -> df with *_z columns appended

Run:  python build_reference.py
"""

import numpy as np
import pandas as pd
import nibabel as nib
import glob, os, csv, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from totalsegmentator.mask_metrics import volume_metrics, shape_metrics, intensity_metrics

# ============================ CONFIG ============================
DATASET  = r"C:\Users\ansar\Downloads\Totalsegmentator_dataset_v201"
OUT_CSV  = r"C:\Users\ansar\Algoverse\report_ct\reference_stats.csv"

IMAGE_NAME = "ct.nii.gz"
SPLIT      = "train"   # MUST be train - val/test stay clean for the classifier
LIMIT      = 200       # subjects to build the reference from; None = all 1082
WORKERS    = 10         # parallel subjects; set 1 to disable multiprocessing
MIN_VOXELS = 20        # ignore FOV slivers when building the reference
MIN_N      = 10        # warn about organs with fewer than this many observations
# ================================================================

# Features that are roughly log-normal: take log1p before computing statistics, so the
# z-score is symmetric instead of being dragged around by the long right tail.
LOG_FEATURES = {"num_voxels", "volume_mm3", "bbox_volume_mm3"}

# Features worth normalising per organ. Deliberately excludes num_components and
# largest_component_fraction: on ground truth those are ~1 for every structure, so a
# reference is meaningless, and connected-component labelling is by far the slowest
# metric to compute. Those stay raw in the classifier, where they are already
# interpretable without organ context.
REFERENCE_FEATURES = [
    "num_voxels", "volume_mm3",
    "centroid_x_rel", "centroid_y_rel", "centroid_z_rel",
    "bbox_x_rel", "bbox_y_rel", "bbox_z_rel",
    "bbox_volume_mm3", "mask_to_bbox_ratio",
    "mean_HU", "median_HU", "std_HU", "p05_HU", "p95_HU",
]


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
    if not subjects:
        subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dataset, "s0*"))
                          if os.path.isdir(p))
    return subjects[:limit] if limit else subjects


# ---------------------------------------------------------------- per subject

def measure_subject(args):
    """Return [(organ, {feature: value}), ...] for one subject's ground-truth masks.

    Module-level and picklable so it works with ProcessPoolExecutor on Windows.
    """
    dataset, subj, image_name, min_voxels = args
    gt_dir   = os.path.join(dataset, subj, "segmentations")
    img_path = os.path.join(dataset, subj, image_name)
    if not (os.path.isdir(gt_dir) and os.path.exists(img_path)):
        return subj, []

    try:
        ct_img = nib.as_closest_canonical(nib.load(img_path))
        ct = ct_img.get_fdata()
        spacing = ct_img.header.get_zooms()[:3]
    except Exception as e:
        return subj, [("__error__", {"msg": f"ct load failed: {e}"})]

    out = []
    for f in sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz"))):
        organ = os.path.basename(f)[:-len(".nii.gz")]
        try:
            m_img = nib.as_closest_canonical(nib.load(f))
            if m_img.shape != ct_img.shape:
                continue
            mask = np.asarray(m_img.dataobj) > 0.5
        except Exception:
            continue
        if int(mask.sum()) < min_voxels:
            continue      # out of field of view, or a sliver: not a reference example
        rec = {}
        rec.update(volume_metrics(mask, spacing))
        rec.update(shape_metrics(mask, spacing))
        rec.update(intensity_metrics(mask, ct))
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
            rows.append({"organ": organ, "feature": feat, "n": len(arr),
                         "median": round(float(med), 6),
                         "q25": round(float(q25), 6),
                         "q75": round(float(q75), 6),
                         "iqr": round(float(q75 - q25), 6),
                         "log_space": int(log_space)})
    return rows


# ---------------------------------------------------------------- apply

def load_reference(path):
    """reference_stats.csv -> {organ: {feature: {median, iqr, log_space}}}"""
    df = pd.read_csv(path)
    ref = {}
    for r in df.itertuples(index=False):
        ref.setdefault(r.organ, {})[r.feature] = {
            "median": r.median, "iqr": r.iqr, "log_space": bool(r.log_space)}
    return ref


def add_relative_features(df, ref, suffix="_z"):
    """Append organ-relative robust z-scores to a metrics dataframe.

    z = (x - organ_median) / (organ_IQR / 1.349)

    The 1.349 converts an IQR into a standard-deviation equivalent for a normal
    distribution, so the result reads on the familiar "how many SDs off" scale while
    staying robust to the outliers that raw std would chase.

    Raw columns are KEPT, not replaced. Trees cope fine with the redundancy, and some
    raw values carry information the z-score throws away.

    Organs absent from the reference table get NaN, which the imputer in the model
    pipeline already handles.
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
            sel = (out["organ"] == organ).to_numpy()
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
    subjects = get_subjects(DATASET, SPLIT, LIMIT)
    print(f"Building reference from {len(subjects)} subject(s), split={SPLIT}")
    print(f"(ground-truth masks only, no segmentation, no GPU)\n")

    observations = {}
    def absorb(subj, recs):
        for organ, rec in recs:
            if organ == "__error__":
                print(f"  {subj}: {rec['msg']}")
                continue
            slot = observations.setdefault(organ, {})
            for k, v in rec.items():
                slot.setdefault(k, []).append(v)

    args = [(DATASET, s, IMAGE_NAME, MIN_VOXELS) for s in subjects]
    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(measure_subject, a): a[1] for a in args}
            for i, fut in enumerate(as_completed(futures), 1):
                subj, recs = fut.result()
                absorb(subj, recs)
                print(f"[{i}/{len(subjects)}] {subj}: {len(recs)} structures")
    else:
        for i, a in enumerate(args, 1):
            subj, recs = measure_subject(a)
            absorb(subj, recs)
            print(f"[{i}/{len(subjects)}] {subj}: {len(recs)} structures")

    rows = build_table(observations)
    if not rows:
        raise SystemExit("No observations collected - check DATASET path.")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    counts = {o: max((len(v) for v in observations[o].values()), default=0)
              for o in observations}
    thin = sorted((n, o) for o, n in counts.items() if n < MIN_N)
    print(f"\n{len(counts)} structures, {len(rows)} organ-feature reference entries")
    print(f"median observations per structure: "
          f"{int(np.median(list(counts.values())))}")
    if thin:
        print(f"\n{len(thin)} structures with fewer than {MIN_N} observations "
              f"(their z-scores will be noisy):")
        for n, o in thin[:15]:
            print(f"  {o:35s} n={n}")
    print(f"\nwritten -> {OUT_CSV}")


if __name__ == "__main__":
    main()