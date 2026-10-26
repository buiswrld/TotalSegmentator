"""
Streams individual subjects out of the public TotalSegmentator CT/MR dataset zip archives
(hosted on Zenodo) via HTTP range requests, so a shared machine never needs the full
23.6GB (CT) / 5.1GB (MR) dataset downloaded or unzipped to disk - only the current
subject's own files are ever staged locally, deleted immediately after that subject is
scored.

Wraps existing stage 1 (run_inference.py) + stage 2 (compute_metrics.py) UNCHANGED: for
each subject, extracts its image + segmentations/ ground truth into a throwaway
single-subject staging directory shaped exactly like a real --dataset-dir (one subject
folder + a one-row meta.csv), invokes the two stages as subprocesses, appends
compute_metrics.py's one-subject output rows onto this shard's running metrics CSV, then
deletes the staging directory (raw image + GT + predictions) before moving to the next
subject.

--num-shards/--shard-index mirror run_inference.py's sharding flags for splitting
subjects across parallel processes (one per GPU); each shard writes its own
combined_metrics_shard{i}.csv (concurrent processes must not share one output file).
Run with --merge-only afterwards to concatenate all shards into one combined_metrics.csv.

Requires: pip install remotezip

Run (one shard):
  python experiments/pipeline/stream_and_process.py \
    --zip-url "https://zenodo.org/records/14710732/files/TotalsegmentatorMRI_dataset_v200.zip?download=1" \
    --modality mr --split test --output-dir <run-dir> --device gpu:0 --num-shards 8 --shard-index 0

Merge after all shards finish:
  python experiments/pipeline/stream_and_process.py --merge-only --output-dir <run-dir> --num-shards 8
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.pipeline.common import find_columns, write_manifest

try:
    from remotezip import RemoteZip
except ImportError:
    raise SystemExit("This script needs the 'remotezip' package: pip install remotezip")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- remote zip helpers

def find_root_prefix(names):
    """The zip's internal top-level folder (e.g. 'TotalsegmentatorMRI_dataset_v200/')."""
    for n in names:
        if n.endswith("meta.csv"):
            return n[: -len("meta.csv")]
    raise SystemExit("Could not find meta.csv inside the zip - is --zip-url correct?")


def load_remote_meta(zf, prefix):
    raw = zf.read(prefix + "meta.csv").decode("utf-8-sig")
    delim = ";" if raw.count(";") > raw.count(",") else ","
    reader = csv.DictReader(io.StringIO(raw), delimiter=delim)
    return list(reader), delim


def stream_subject(zf, prefix, subject, dest_dataset_dir):
    """Extract only this subject's files (image + segmentations/) via range requests."""
    subj_prefix = f"{prefix}{subject}/"
    members = [n for n in zf.namelist() if n.startswith(subj_prefix) and not n.endswith("/")]
    if not members:
        return False
    for member in members:
        rel = member[len(subj_prefix):]
        dest_path = os.path.join(dest_dataset_dir, subject, rel)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with zf.open(member) as src, open(dest_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return True


def write_stub_meta(dest_dataset_dir, header, row, delim):
    with open(os.path.join(dest_dataset_dir, "meta.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter=delim)
        w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- per-subject pipeline

def run(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def append_rows(src_csv, dest_csv):
    with open(src_csv) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames
    write_header = not os.path.exists(dest_csv)
    with open(dest_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def process_subject(zf, prefix, subj, args, meta_by_id, header, delim, combined_csv, keep_staging=False):
    staging = tempfile.mkdtemp(prefix=f"ts_stream_{subj}_")
    dataset_dir = os.path.join(staging, "dataset")
    predictions_dir = os.path.join(staging, "predictions")
    os.makedirs(dataset_dir, exist_ok=True)
    timings = {}
    try:
        t0 = time.time()
        if not stream_subject(zf, prefix, subj, dataset_dir):
            return False, "not found in zip", timings
        write_stub_meta(dataset_dir, header, meta_by_id[subj], delim)
        timings["stream_s"] = round(time.time() - t0, 1)

        extra = [f"--extra-arg={e}" for e in args.extra_args]
        t0 = time.time()
        run([sys.executable, os.path.join(THIS_DIR, "run_inference.py"),
             "--dataset-dir", dataset_dir, "--predictions-dir", predictions_dir,
             "--modality", args.modality, "--split", args.split]
            + (["--device", args.device] if args.device else [])
            + extra)
        timings["inference_subprocess_s"] = round(time.time() - t0, 1)

        subj_metrics_csv = os.path.join(staging, "combined_metrics.csv")
        t0 = time.time()
        run([sys.executable, os.path.join(THIS_DIR, "compute_metrics.py"),
             "--dataset-dir", dataset_dir, "--predictions-dir", predictions_dir,
             "--modality", args.modality, "--split", args.split,
             "--output-csv", subj_metrics_csv, "--iou-accept", str(args.iou_accept)])
        timings["metrics_subprocess_s"] = round(time.time() - t0, 1)

        append_rows(subj_metrics_csv, combined_csv)
        return True, None, timings
    except subprocess.CalledProcessError as e:
        return False, str(e), timings
    finally:
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------- merge

def merge_shards(output_dir, num_shards):
    dest = os.path.join(output_dir, "combined_metrics.csv")
    if os.path.exists(dest):
        os.remove(dest)
    total = 0
    for i in range(num_shards):
        shard_csv = os.path.join(output_dir, f"combined_metrics_shard{i}.csv")
        if not os.path.exists(shard_csv):
            print(f"  WARNING: {shard_csv} missing - shard {i} may not have finished.")
            continue
        with open(shard_csv) as fh:
            n = sum(1 for _ in fh) - 1
        append_rows(shard_csv, dest)
        total += n
        print(f"  shard {i}: {n} rows")
    print(f"\n{total} total rows merged -> {dest}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-url", default=None, help="Public Zenodo (or similar range-request-capable) zip URL for the dataset.")
    parser.add_argument("--modality", default=None, choices=["ct", "mr"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", required=True, help="Where the accumulated metrics CSV(s) go.")
    parser.add_argument("--device", default=None, help="Passed through to run_inference.py (e.g. gpu:0).")
    parser.add_argument("--iou-accept", type=float, default=0.90)
    parser.add_argument("--extra-arg", action="append", default=[], dest="extra_args",
                        help="Extra TotalSegmentator CLI flag, passed through to run_inference.py, repeatable.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--keep-staging", action="store_true",
                        help="Don't delete each subject's staged files after scoring (debugging only - uses much more disk).")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip streaming; just concatenate combined_metrics_shard*.csv in --output-dir into combined_metrics.csv.")
    args = parser.parse_args()

    if args.merge_only:
        merge_shards(args.output_dir, args.num_shards)
        return

    if not args.zip_url or not args.modality:
        raise SystemExit("--zip-url and --modality are required unless --merge-only is set.")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}) but got {args.shard_index}")

    os.makedirs(args.output_dir, exist_ok=True)
    combined_csv_name = (f"combined_metrics_shard{args.shard_index}.csv"
                         if args.num_shards > 1 else "combined_metrics.csv")
    combined_csv = os.path.join(args.output_dir, combined_csv_name)

    with RemoteZip(args.zip_url) as zf:
        names = zf.namelist()
        prefix = find_root_prefix(names)
        meta_rows, delim = load_remote_meta(zf, prefix)
        id_col, split_col = find_columns(list(meta_rows[0].keys()))
        subjects = [r[id_col] for r in meta_rows if r[split_col].strip().lower() == args.split.lower()]
        if args.limit:
            subjects = subjects[: args.limit]
        if args.num_shards > 1:
            subjects = subjects[args.shard_index::args.num_shards]
        shard_tag = f" | shard {args.shard_index}/{args.num_shards}" if args.num_shards > 1 else ""
        print(f"{len(subjects)} subject(s) to stream+process | modality={args.modality} "
              f"split={args.split}{shard_tag}\n")

        meta_by_id = {r[id_col]: r for r in meta_rows}
        header = list(meta_rows[0].keys())

        done, failed = [], []
        for i, subj in enumerate(subjects, 1):
            t0 = time.time()
            ok, err, timings = process_subject(zf, prefix, subj, args, meta_by_id, header, delim,
                                               combined_csv, keep_staging=args.keep_staging)
            elapsed = time.time() - t0
            breakdown = " | ".join(f"{k}={v}s" for k, v in timings.items())
            if ok:
                done.append(subj)
                print(f"[{i}/{len(subjects)}] {subj}: done in {elapsed:.1f}s  ({breakdown})")
            else:
                failed.append(subj)
                print(f"[{i}/{len(subjects)}] {subj}: FAILED ({err})  ({breakdown})")

    manifest_name = (f"stream_manifest_shard{args.shard_index}.json"
                     if args.num_shards > 1 else "stream_manifest.json")
    write_manifest(
        os.path.join(args.output_dir, manifest_name),
        stage="stream_and_process", zip_url=args.zip_url, modality=args.modality, split=args.split,
        limit=args.limit, num_shards=args.num_shards, shard_index=args.shard_index,
        n_subjects=len(subjects), n_done=len(done), n_failed=len(failed), failed_subjects=failed,
    )
    print(f"\n{len(done)} done, {len(failed)} failed. {combined_csv_name} -> {args.output_dir}")


if __name__ == "__main__":
    main()
