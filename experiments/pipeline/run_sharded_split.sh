#!/bin/bash
# Stage 1, sharded across N concurrent processes on a SINGLE GPU (not one process per
# GPU - one GPU running N processes at once). A single subject's nnU-Net inference
# leaves the GPU mostly idle between sliding-window patches, so running several subjects
# concurrently on the same card gives real throughput gains (measured on an A100: ~6.5x
# vs. sequential at 8-way concurrency, see remote-gpu-inference branch history).
#
# Resumable by construction: relies entirely on run_inference.py's own per-subject skip
# logic (a subject with existing prediction files is skipped). Safe to Ctrl-C or lose the
# session and rerun this exact command later - already-finished subjects across all
# shards are picked up automatically, nothing is redone.
#
# Backgrounds all shards and waits for them within this script, printing each shard's
# tail when done. To survive a disconnected notebook/terminal for a long real run, wrap
# the WHOLE script invocation (not individual shards) with nohup + disown:
#   nohup experiments/pipeline/run_sharded_split.sh ... > /tmp/split_run.log 2>&1 &
#   disown
#   tail -f /tmp/split_run.log
#
# Usage:
#   run_sharded_split.sh <dataset-dir> <modality> <split> <predictions-dir> [num-shards] [device] [extra run_inference.py args...]
#
# Example (8-way, full split):
#   experiments/pipeline/run_sharded_split.sh /opt/dlami/nvme/datasets/mri mr train /opt/dlami/nvme/runs/mri_full/predictions/train 8 gpu:0
#
# Example (small mimic subset):
#   experiments/pipeline/run_sharded_split.sh /opt/dlami/nvme/datasets/mri mr train /opt/dlami/nvme/runs/mri_smoke/predictions/train 8 gpu:0 --limit 24

set -e
DATASET_DIR=$1
MODALITY=$2
SPLIT=$3
PRED_DIR=$4
NUM_SHARDS=${5:-8}
DEVICE=${6:-gpu:0}
EXTRA_ARGS=("${@:7}")

if [ -z "$PRED_DIR" ]; then
  echo "Usage: $0 <dataset-dir> <modality> <split> <predictions-dir> [num-shards] [device] [extra args...]"
  exit 2
fi

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PRED_DIR/_shard_logs"
mkdir -p "$PRED_DIR" "$LOG_DIR"

echo "Launching $NUM_SHARDS shard(s) | modality=$MODALITY split=$SPLIT device=$DEVICE -> $PRED_DIR"
PIDS=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
  python "$THIS_DIR/run_inference.py" \
    --dataset-dir "$DATASET_DIR" --predictions-dir "$PRED_DIR" \
    --modality "$MODALITY" --split "$SPLIT" --device "$DEVICE" \
    --num-shards "$NUM_SHARDS" --shard-index "$i" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/shard_${i}.log" 2>&1 &
  PIDS+=($!)
  echo "  shard $i -> PID $!"
done
echo "${PIDS[@]}" > "$LOG_DIR/pids.txt"

echo "Waiting for all shards to finish (safe to Ctrl-C and rerun this script later)..."
time wait "${PIDS[@]}"

echo ""
echo "All shards finished. Per-shard tail:"
for i in $(seq 0 $((NUM_SHARDS-1))); do
  echo "--- shard $i ---"
  tail -4 "$LOG_DIR/shard_${i}.log"
done
