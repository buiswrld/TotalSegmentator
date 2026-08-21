#!/bin/bash
# Runs stage 1 (sharded, GPU, via run_sharded_split.sh) then stage 2 (compute_metrics.py,
# CPU) for ONE dataset+subject-selection, then copies just the lightweight stage-2 output
# into the git-tracked run directory and commits+pushes it.
#
# Why: on a time-boxed GPU grant (e.g. a shared box wiped at a hard deadline with no
# recovery), raw predictions are large, gitignored, and fully reproducible from code +
# weights - not worth protecting. What IS worth protecting is the hours of GPU time spent
# producing them, which this captures by compacting predictions into a small metrics CSV
# and pushing it immediately, so a wipe right after never loses more than the piece
# currently in flight. The scratch run directory (predictions, full metrics duplicate)
# stays on local/scratch disk and is never copied into the repo.
#
# ROLE-NAME is what output directories/checkpoint commits are named (e.g.
# "classifier_train", "classifier_test", "train") - kept separate from SPLIT (the actual
# --split value passed to run_inference.py/compute_metrics.py, e.g. "all") because
# --split all is deterministic-but-identical across different --offset/--limit windows,
# so the split name alone isn't distinctive enough to use as a directory/commit label.
#
# Usage:
#   checkpoint_split.sh <dataset-dir> <modality> <split> <role-name> <scratch-run-dir> <repo-run-name> [num-shards] [device] [-- extra args for run_inference.py/compute_metrics.py]
#
# Example (named split, unchanged from before):
#   experiments/pipeline/checkpoint_split.sh /opt/dlami/nvme/datasets/mri mr train train /opt/dlami/nvme/runs/mri_full mri_full_remote 8 gpu:0
#
# Example (pooled, offset-sliced):
#   experiments/pipeline/checkpoint_split.sh /opt/dlami/nvme/datasets/mri mr all classifier_train /opt/dlami/nvme/runs/mri_full mri_full_remote 8 gpu:0 -- --offset 0 --limit 493

set -e
DATASET_DIR=$1
MODALITY=$2
SPLIT=$3
ROLE_NAME=$4
SCRATCH_RUN_DIR=$5
REPO_RUN_NAME=$6
NUM_SHARDS=${7:-8}
DEVICE=${8:-gpu:0}
shift 8 2>/dev/null || shift $#
if [ "$1" = "--" ]; then shift; fi
EXTRA_ARGS=("$@")

if [ -z "$REPO_RUN_NAME" ]; then
  echo "Usage: $0 <dataset-dir> <modality> <split> <role-name> <scratch-run-dir> <repo-run-name> [num-shards] [device] [-- extra args]"
  exit 2
fi

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_RUN_DIR="$REPO_ROOT/experiments/eval_runs/$REPO_RUN_NAME"

echo "=== stage 1: sharded inference for $MODALITY/$ROLE_NAME (--split $SPLIT ${EXTRA_ARGS[*]}) ==="
"$THIS_DIR/run_sharded_split.sh" "$DATASET_DIR" "$MODALITY" "$SPLIT" \
  "$SCRATCH_RUN_DIR/predictions/$ROLE_NAME" "$NUM_SHARDS" "$DEVICE" "${EXTRA_ARGS[@]}"

echo ""
echo "=== stage 2: compute_metrics for $MODALITY/$ROLE_NAME ==="
mkdir -p "$SCRATCH_RUN_DIR/metrics/$ROLE_NAME"
python "$THIS_DIR/compute_metrics.py" \
  --dataset-dir "$DATASET_DIR" --predictions-dir "$SCRATCH_RUN_DIR/predictions/$ROLE_NAME" \
  --modality "$MODALITY" --split "$SPLIT" "${EXTRA_ARGS[@]}" \
  --output-csv "$SCRATCH_RUN_DIR/metrics/$ROLE_NAME/combined_metrics.csv"

echo ""
echo "=== checkpoint: copying lightweight outputs into the repo and pushing ==="
mkdir -p "$REPO_RUN_DIR/metrics/$ROLE_NAME"
cp "$SCRATCH_RUN_DIR/metrics/$ROLE_NAME"/*.csv "$REPO_RUN_DIR/metrics/$ROLE_NAME/" 2>/dev/null || true
cp "$SCRATCH_RUN_DIR/metrics/$ROLE_NAME"/*.json "$REPO_RUN_DIR/metrics/$ROLE_NAME/" 2>/dev/null || true

cd "$REPO_ROOT"
git add "experiments/eval_runs/$REPO_RUN_NAME/metrics/$ROLE_NAME"
if git diff --cached --quiet; then
  echo "nothing new to commit (metrics unchanged)"
else
  git commit -m "checkpoint: $REPO_RUN_NAME $MODALITY/$ROLE_NAME metrics ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  git push
fi

echo ""
echo "Checkpoint complete for $MODALITY/$ROLE_NAME. Predictions remain on scratch disk at $SCRATCH_RUN_DIR/predictions/$ROLE_NAME (not copied - large + reproducible)."
