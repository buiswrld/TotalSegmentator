# Remote GPU box runbook — operational notes from the first full run

This is an operational knowledge base for running the QC pipeline's stage 1 (inference)
on a shared, time-boxed remote GPU grant (JupyterHub, multi-tenant A100 x8 box). It
documents what actually happened during the first full pooled classifier-pool run —
what failed, why, and what fixed it — so a future agent (or human) doesn't have to
rediscover the same things. This is separate from RESEARCH.md (methodology/findings)
and README.md (day-to-day pipeline usage); this file is specifically about the mechanics
of running on this kind of environment.

## The environment

- JupyterHub-provisioned box, `nvidia-smi` shows 8x A100-SXM4-40GB, but a normal session
  only sees **1 GPU** by default (`CUDA_VISIBLE_DEVICES` set to a single index). It's a
  soft env var, not a hard container restriction - overriding it to all 8 indices does
  work - but the box is genuinely **shared with other tenants** whose jobs occupy the
  other GPUs unpredictably, so grabbing more than your allocated share is not appropriate
  even though nothing technically stops you.
- 96 CPU cores, ~1.1TiB system RAM - CPU/RAM were never the bottleneck; GPU compute and
  GPU *memory* were.
- Large local scratch disk at `/opt/dlami/nvme` (several TB free) - use this for datasets
  and run directories, not the home directory's smaller root disk.
- **Time-boxed grant with a hard wipe deadline** - everything on the box is deleted with
  no recovery when the grant expires. This is why the pipeline checkpoints per-piece
  instead of running everything then saving once at the end (see `checkpoint_split.sh`).

## Environment setup gotchas (all one-time fixes)

- `pip install -e .` puts console scripts (`totalseg_info`, etc.) in `~/.local/bin`,
  which isn't on `PATH` by default - `export PATH="$HOME/.local/bin:$PATH"`.
- The box's pre-installed `torch`/`torchvision` (e.g. 2.13.0/0.28.0) will NOT match a
  fresh `pip install torch==2.6.0` you do for this project - installing a different
  torch version without also matching torchvision breaks torchvision's compiled ops
  (`RuntimeError: operator torchvision::nms does not exist`), which surfaces obliquely
  because nnU-Net's trainer auto-discovery unconditionally imports every trainer file
  (including an unrelated "Primus" architecture) that happens to import torchvision
  transitively. Fix: install the torchvision build that matches whatever torch version
  you installed (e.g. `torchvision==0.21.0` for `torch==2.6.0`), same `--index-url`.
- `total`/`total_mr` (what this pipeline uses) are NOT licensed tasks - confirmed via
  `totalseg_info --list-tasks`, no `totalseg_set_license` needed.

## Data: stream vs. bulk download

- Started with a per-subject streaming approach (`stream_and_process.py`, pulls
  individual files out of the public Zenodo zip via HTTP range requests) specifically to
  avoid downloading the full dataset onto a shared box - a prior team downloading 1.5TB
  had broken the box for everyone, and the admin's policy was "stream, or ask first for
  anything large." Streaming works but 3+ concurrent processes hitting the same Zenodo
  file trip a 429 rate limit.
- Got explicit admin approval to bulk-download instead (28.7GB total vs. TB-scale disk
  free - not remotely close to the incident that prompted the policy). **If doing this
  again on a similar shared-policy box: always get explicit sign-off before bulk
  downloading, even when the math clearly says it's fine** - the policy exists because a
  previous incident, not because of the raw byte count.
- **Both dataset zips have NO top-level wrapping folder** - `unzip`ing them into the same
  destination directory silently interleaves MR and CT subject folders, which use the
  *same* `sXXXX` ID scheme for different patients. This caused real, silent data loss
  (subjects overwritten) the first time - **always `unzip -d <separate-dir>` per
  dataset**, never into a shared parent directory.
- Zenodo download speed on this box was wildly inconsistent (1.2MB/s at worst, fine at
  best) - not obviously explainable, possibly transient. If it's slow, retry once before
  assuming something's fundamentally wrong.

## GPU concurrency: same-GPU sharding, not multi-GPU

Since only 1 GPU is actually usable per session, "sharding" here means running N
*concurrent processes on the same GPU*, not N processes across N GPUs. This works because
a single subject's nnU-Net inference leaves the GPU mostly idle between sliding-window
patches (confirmed directly with `nvidia-smi dmon` - a lone job showed spiky 0-36%
utilization, not saturated).

Measured on this box (MR, full resolution, no fast mode):
| concurrency | effective time/subject |
|---|---|
| sequential | ~44.7s |
| 3-way | ~15.4s |
| 4-way | ~11.55s |
| 5-way | ~9.30s |
| 6-way | ~7.98s |
| 8-way | ~6.48s (but this is where it broke - see below) |

**This is NOT free scaling forever - there is a real ceiling, and it's GPU memory, not
GPU compute.** At 8-way concurrency, once other tenants' usage grew to ~29GB of the 40GB
card, our own shards started hitting `CUDA OutOfMemoryError`. Individual shard peak
memory varies a lot by subject (roughly 0.6GB-4.6GB seen in practice) - there is no fixed
"safe" concurrency number, because it depends on what else is running on the shared card
*right now*. **Check `nvidia-smi --query-gpu=memory.used,memory.total` before choosing a
concurrency level, and be willing to drop it (we went 8 -> 2 -> 4) if you see OOM.**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (suggested by the OOM error itself)
was added as a mitigation but doesn't remove the fundamental memory ceiling.

## The silent-hang failure mode (important - looks like "stuck", is actually "crashed")

When a shard hits `CUDA OutOfMemoryError`, the actual crash happens inside nnU-Net's
internal data-loading **background thread**, not the main thread. Python does not
propagate an unhandled exception in a background thread to the main process - the main
thread just blocks forever waiting on a queue that will never receive anything again.
Symptoms: the process is still alive in `ps aux`, using ~0.0% CPU, `ELAPSED` time growing
indefinitely, **no error visible anywhere in `ps`/`top`/the top-level log**. The only way
to see the real error is checking that specific shard's own log file (each shard managed
by `run_sharded_split.sh` writes to `<predictions-dir>/_shard_logs/shard_<i>.log`) - grep
it for `Error`/`Traceback`/`Exception`, don't just `tail` the last few lines, since the
crash can be buried above whatever else got printed after.

**Diagnostic checklist when a shard looks stuck:**
1. `ps -o pid,pcpu,etime,stat -p <pid>` - 0.0% CPU for a long `ELAPSED` = hung, not slow.
2. Check that shard's own log in `_shard_logs/` for the real error.
3. `nvidia-smi` memory usage - if near-full, OOM is the likely cause.
4. If nothing in the shard log, check `free -h` and `dmesg | grep -i oom` for a
   system-level (not GPU) OOM kill - ruled this out once (1TiB RAM, no kernel OOM
   activity), but worth checking before assuming it's the same GPU-memory cause every
   time.

## Process detachment: `nohup ... & disown` was NOT enough on this box

Long jobs need to survive terminal disconnects. The standard `nohup cmd > log 2>&1 &`
plus `disown` pattern, which normally works fine, **did not survive a `Ctrl-C` typed
into the same terminal** on this JupyterHub setup - even though the job was
backgrounded and disowned, `Ctrl-C` on a foreground `tail -f` in the same terminal killed
the whole background job too (likely a process-group quirk of this particular pty/job
control setup, not standard bash behavior). This caused real, silent, undetected job
loss more than once.

**Fix: use `setsid nohup bash -c '...' > log 2>&1 < /dev/null & disown` instead** - `setsid`
starts a genuinely new session, fully detached from the controlling terminal, which
survived subsequent testing. Also: **never `Ctrl-C` in the same terminal that launched a
background job you care about** - if you want to watch a log live, open a separate
terminal tab for `tail -f` and only `Ctrl-C` that one. Prefer non-blocking status checks
(`tail -n 40 log`, `ps aux | grep ...`) over `tail -f` when in doubt.

Even `setsid` isn't bulletproof against everything: a job disappeared once with **no
error anywhere** (no OOM, no exception, no kernel OOM-killer activity, system RAM mostly
free) after a long idle period with no notebook/terminal interaction. Best working
hypothesis: **JupyterHub's idle-server culler** stopped the whole session/container,
which kills everything inside regardless of process-level detachment - there's no
process-level trick that survives the entire container being torn down. If this is a
recurring problem, ask the platform admin about idle-timeout policy, and/or periodically
touch the notebook/terminal to avoid triggering it.

## Progress tracking: folder existence != a finished prediction

`run_inference.py`'s resumability creates a subject's output folder *before* running
TotalSegmentator (so the "already had predictions" check has somewhere to look), then
writes `.nii.gz` files into it. If a shard gets killed mid-write (e.g. from a `pkill -9`
during recovery from one of the above failure modes), it leaves an **empty stub folder**
behind. A progress check that just counts folders (`find ... -type d | wc -l`)
overcounts - it'll include these empty stubs as "done" when they aren't.

**Correct progress check** counts folders that actually contain `.nii.gz` files:
```bash
count=0
for d in <predictions-dir>/s*/; do
  compgen -G "${d}*.nii.gz" > /dev/null && count=$((count+1))
done
echo "$count / <expected total>"
```
This isn't a real problem for correctness - `run_inference.py`'s own skip-logic checks
for the actual files, so an empty stub gets correctly retried on the next pass - it's
only a problem for *knowing how much progress has actually been made* if you're using
the naive folder-count method.

## Architecture: pooled, split-label-independent classifier train/test

The QC classifier is downstream of a *frozen* TotalSegmentator and has no reason to
respect TotalSegmentator's own train/val/test split boundaries (those exist to evaluate
TotalSegmentator itself). `common.py::get_subjects(dataset_dir, split, limit, offset)`
supports `split="all"`, which pools every subject regardless of original label,
deterministically shuffles (fixed seed, `POOLED_SHUFFLE_SEED = 0`), then `offset`/`limit`
carve out a window. Reference table (ground-truth-only, no GPU) is built from **every**
subject; classifier train/test pools are **also** built from every subject, split 80:20 -
deliberately allowing overlap between the reference population and classifier
examples (see RESEARCH.md section 2a for the full reasoning and the accepted trade-off).

`checkpoint_split.sh` takes a `ROLE_NAME` distinct from `SPLIT` specifically because
`--split all` is identical across different `--offset`/`--limit` windows (e.g.
`classifier_train` and `classifier_test` are both `--split all`, just different offsets)
- the split value alone isn't distinctive enough to use as an output directory/checkpoint
name.

## Checkpointing strategy

Per-piece, not per-run: `checkpoint_split.sh` runs stage 1 (sharded inference) then stage
2 (metrics) for ONE piece (e.g. `mr classifier_train`), then immediately copies just the
resulting `combined_metrics.csv` (small, a few MB) into the git-tracked
`experiments/eval_runs/<run-name>/metrics/<role>/` and commits+pushes. Raw predictions
(large, reproducible from code+weights) stay on scratch disk and are never pushed. This
means the worst case from any of the failures above is losing whatever single piece was
mid-flight - everything already checkpointed is permanently safe regardless of what
happens to the box afterward.

## Known stale/leftover data

`experiments/eval_runs/mri_full_remote/metrics/train/` (note: `train`, not
`classifier_train`) is leftover from an early, architecturally-superseded attempt that
used named splits instead of the pooled scheme, and happened to finish + checkpoint
itself before being killed. It's harmless but unused - don't confuse it with
`metrics/classifier_train/`, the correct pooled-scheme output. Left in place rather than
cleaned up (consistent with how `mr_full_run`'s pre-fix leakage issue was handled -
caveat and move on, not retroactive cleanup).
