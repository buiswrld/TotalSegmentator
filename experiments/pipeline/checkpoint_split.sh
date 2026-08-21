#!/bin/bash
# Runs stage 1 (sharded, GPU, via run_sharded_split.sh) then stage 2 (compute_metrics.py,
# CPU) for ONE dataset+split, then copies just the lightweight stage-2 output into the
# git-tracked run directory and commits+pushes it.
#
# Why: on a time-boxed GPU grant (e.g. a shared box wiped at a hard deadline with no
# recovery), raw predictions are large, gitignored, and fully reproducible from code +
# weights - not worth protecting. What IS worth protecting is the hours of GPU time spent
# producing them, which this captures by compacting predictions into a small metrics CSV
# and pushing it immediately, so a wipe right after never loses more than the split
# currently in flight. The scratch run directory (predictions, full metrics duplicate)
# stays on local/scratch disk and is never copied into the repo.
#
# Usage:
#   checkpoint_split.sh <dataset-dir> <modality> <split> <scratch-run-dir> <repo-run-name> [num-shards] [device]
#
# Example:
#   experiments/pipeline/checkpoint_split.sh /opt/dlami/nvme/datasets/mri mr train /opt/dlami/nvme/runs/mri_full mri_full_remote 8 gpu:0

set -e
DATASET_DIR=$1
MODALITY=$2
SPLIT=$3
SCRATCH_RUN_DIR=$4
REPO_RUN_NAME=$5
NUM_SHARDS=${6:-8}
DEVICE=${7:-gpu:0}

if [ -z "$REPO_RUN_NAME" ]; then
  echo "Usage: $0 <dataset-dir> <modality> <split> <scratch-run-dir> <repo-run-name> [num-shards] [device]"
  exit 2
fi

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_RUN_DIR="$REPO_ROOT/experiments/eval_runs/$REPO_RUN_NAME"

echo "=== stage 1: sharded inference for $MODALITY/$SPLIT ==="
"$THIS_DIR/run_sharded_split.sh" "$DATASET_DIR" "$MODALITY" "$SPLIT" \
  "$SCRATCH_RUN_DIR/predictions/$SPLIT" "$NUM_SHARDS" "$DEVICE"

echo ""
echo "=== stage 2: compute_metrics for $MODALITY/$SPLIT ==="
mkdir -p "$SCRATCH_RUN_DIR/metrics/$SPLIT"
python "$THIS_DIR/compute_metrics.py" \
  --dataset-dir "$DATASET_DIR" --predictions-dir "$SCRATCH_RUN_DIR/predictions/$SPLIT" \
  --modality "$MODALITY" --split "$SPLIT" \
  --output-csv "$SCRATCH_RUN_DIR/metrics/$SPLIT/combined_metrics.csv"

echo ""
echo "=== checkpoint: copying lightweight outputs into the repo and pushing ==="
mkdir -p "$REPO_RUN_DIR/metrics/$SPLIT"
cp "$SCRATCH_RUN_DIR/metrics/$SPLIT"/*.csv "$REPO_RUN_DIR/metrics/$SPLIT/" 2>/dev/null || true
cp "$SCRATCH_RUN_DIR/metrics/$SPLIT"/*.json "$REPO_RUN_DIR/metrics/$SPLIT/" 2>/dev/null || true

cd "$REPO_ROOT"
git add "experiments/eval_runs/$REPO_RUN_NAME/metrics/$SPLIT"
if git diff --cached --quiet; then
  echo "nothing new to commit (metrics unchanged)"
else
  git commit -m "checkpoint: $REPO_RUN_NAME $MODALITY/$SPLIT metrics ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  git push
fi

echo ""
echo "Checkpoint complete for $MODALITY/$SPLIT. Predictions remain on scratch disk at $SCRATCH_RUN_DIR/predictions/$SPLIT (not copied - large + reproducible)."
