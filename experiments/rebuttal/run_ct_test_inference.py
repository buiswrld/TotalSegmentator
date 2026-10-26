"""
Item G: CT held-out test inference (89 subjects) on CPU, for the discussion period /
camera-ready. NOT part of the frozen Phase 3 deliverables (which read no CT test label).

No environment changes: uses the already-installed TotalSegmentator 2.15.0 via its own entry
point module (python -m totalsegmentator.bin.TotalSegmentator), because the editable install
did not create the TotalSegmentator.exe console launcher. This runs the identical main() with
identical defaults to the CLI the dev pipeline used (ensure_predictions):
    TotalSegmentator -i <ct.nii.gz> -o <pred_dir> -ta total -d cpu
i.e. task total, per-structure output (no --ml), full resolution (no --fast). Same flags as
predictions_ct_val. Idempotent / resumable: a subject already having prediction files is
skipped.
"""
import os, sys, glob, json, subprocess, datetime
sys.path.insert(0, ".")
from experiments.pipeline.common import get_subjects

DATASET = os.environ["SSAT_CT_DATASET"]
PRED_DIR = os.environ["SSAT_CT_TEST_PREDICTIONS"]
TASK = "total"
os.makedirs(PRED_DIR, exist_ok=True)

subjects = get_subjects(DATASET, "test", None)
print(f"CT test inference: {len(subjects)} subjects -> {PRED_DIR}", flush=True)

done, skipped, failed = [], [], []
for i, subj in enumerate(subjects, 1):
    ct = os.path.join(DATASET, subj, "ct.nii.gz")
    out = os.path.join(PRED_DIR, subj)
    if glob.glob(os.path.join(out, "*.nii.gz")):
        skipped.append(subj)
        print(f"[{i}/{len(subjects)}] {subj}: already has predictions, skipping", flush=True)
        continue
    if not os.path.exists(ct):
        failed.append((subj, "missing ct.nii.gz"))
        print(f"[{i}/{len(subjects)}] {subj}: MISSING ct.nii.gz, skipping", flush=True)
        continue
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator",
           "-i", ct, "-o", out, "-ta", TASK, "-d", "cpu"]
    print(f"[{i}/{len(subjects)}] {subj}: running ...", flush=True)
    try:
        subprocess.run(cmd, check=True)
        done.append(subj)
    except subprocess.CalledProcessError as e:
        failed.append((subj, f"exit {e.returncode}"))
        print(f"[{i}/{len(subjects)}] {subj}: FAILED ({e.returncode})", flush=True)

manifest = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "stage": "ct_test_inference_for_discussion_period",
    "totalsegmentator_version": "2.15.0",
    "task": TASK, "device": "cpu", "flags": "defaults (no --ml, no --fast); per-structure full-res",
    "invocation": "python -m totalsegmentator.bin.TotalSegmentator -i <ct> -o <out> -ta total -d cpu",
    "dataset_dir": DATASET, "predictions_dir": PRED_DIR, "split": "test",
    "n_subjects": len(subjects), "n_done": len(done), "n_skipped": len(skipped),
    "n_failed": len(failed), "failed": failed,
    "note": "NOT used in the frozen Phase 3 deliverables; same flags as predictions_ct_val.",
}
with open(os.path.join(PRED_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nDONE. completed {len(done)}, skipped {len(skipped)}, failed {len(failed)}", flush=True)
if failed:
    print("failed:", failed, flush=True)
