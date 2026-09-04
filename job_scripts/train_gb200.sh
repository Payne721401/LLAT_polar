#!/bin/bash
# ============================================================================
# LLAT polar - NCHC GB200 (aarch64) training. Companion to train_h200.sh, which
# stays the script for the H200 cluster; the two are not interchangeable.
#
#   sbatch --export=ALL,SAVE_TOP_K=5,FRESH=1,CHAIN=1,RUNDIR=runs/NAME,OVERLAY=experiments/NAME.yaml \
#          job_scripts/train_gb200.sh
#
# Five differences from the H200 script, every one forced by the machine or the
# queue.
#
# 1. Four nodes, not one, and sixteen GPUs rather than eight. QOS p_gb200_r1
#    sets MinTRES gres/gpu=16, so a smaller job is rejected outright with
#    QOSMinGRES however high its priority - the reason a one-GPU job has been
#    sitting at the top of the r2 queue going nowhere. Four GPUs per node makes
#    sixteen GPUs exactly four nodes.
#
# 2. batch_size must be 2, in the overlay. It is per device, so sixteen GPUs at
#    2 reproduces the H200 runs' effective batch of 32. Leaving it at 4 would
#    double the effective batch, halve the per-sample learning rate and make the
#    run a different experiment rather than a faster one.
#
# 3. Segments are short by default, four hours rather than the partition's
#    twenty-four. Slurm's backfill scheduler will start a low-priority job early
#    only if it finishes before the next reserved start, and no gap in this
#    queue is a day wide - a 23:55 job therefore waits for the front of the
#    queue and nothing else. Four hours fits the gaps that open when the
#    scheduler is accumulating nodes for a seven- or eight-node job. Segment
#    length costs nothing scientifically: max_steps is absolute, the learning
#    rate is constant after warmup, and the script resumes automatically.
#
# 4. aarch64. The cluster's miniconda3 is an x86-64 build and cannot execute on
#    these nodes at all ("cannot execute binary file"), and PyTorch stopped
#    publishing conda packages after 2.5 while Blackwell needs >= 2.7. So there
#    is no conda here: the environment is a venv built with /usr/bin/python3.11
#    and pip, and the CUDA runtime comes from the wheels rather than a module.
#    --export=ALL still drags the login node's x86 conda in, which is why the
#    environment is stripped below before the venv is sourced.
#
# 5. MaxTRESPA gres/gpu=32 is per ACCOUNT, so two four-node jobs are the whole
#    of mst115002's r1 allowance and other members of the account compete for
#    it. Check `squeue -A mst115002 -p gb200-r1` before assuming both halves of
#    a comparison can run at once.
#
# Checkpoints: /home is 100 GB with ~45 free and one checkpoint is 305 MB, so
# the default save_top_k of 30 is 9.2 GB per run and a full quota kills the job
# mid-write with an error that never mentions disks. Set SAVE_TOP_K.
# ============================================================================
#SBATCH --account=MST115002
#SBATCH --job-name=LLAT_gb200
#SBATCH --partition=gb200-r1
#SBATCH --nodes=4                     # 16 GPUs: the QOS minimum, see note 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4           # one task per GPU (DDP)
#SBATCH --cpus-per-task=32            # 136 effective cores / 4 tasks, with headroom
#SBATCH --time=04:00:00               # short on purpose, see note 3; override with -t
#SBATCH --output=job_logs/job-%j.out
#SBATCH --error=job_logs/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=90727sam@gmail.com

set -eo pipefail          # not -u: an empty EXTRA array is expanded below
mkdir -p job_logs

echo "=========================================="
date
echo "Job     : ${SLURM_JOB_ID:-NA}"
echo "Nodes   : ${SLURM_JOB_NODELIST:-NA}"
echo "Tasks   : ${SLURM_NTASKS:-NA}  (nodes ${SLURM_JOB_NUM_NODES:-NA})"
echo "WorkDir : $(pwd)"
echo "Arch    : $(uname -m)"

[ "$(uname -m)" = "aarch64" ] || { echo "!!! not on an aarch64 node - wrong partition?"; exit 1; }

RUNDIR="${RUNDIR:-/work/$USER/LLAT_polar_runs/default}"
mkdir -p "$RUNDIR"
echo "RunDir  : $RUNDIR"

# ---- Environment ----------------------------------------------------------
# Strip the x86 conda that --export=ALL carried over from the login node. Its
# python cannot run here, so leaving it on PATH is not merely untidy: whichever
# of the two `python` entries wins decides whether the job starts at all.
unset PYTHONPATH PYTHONHOME
while [ -n "${CONDA_PREFIX:-}" ]; do conda deactivate 2>/dev/null || break; done
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -e miniconda -e anaconda | paste -sd:)
export PATH

VENV="${VENV:-$HOME/venv-aarch64}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "--- GPU ---"
nvidia-smi -L
echo "--- Python ---"
echo "VIRTUAL_ENV = ${VIRTUAL_ENV:-<empty>}"
echo "python      = $(command -v python)"
python -c "import platform, torch; print('arch', platform.machine(), '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count(), '| cap', torch.cuda.get_device_capability())" || {
    echo "!!! venv not usable. Rebuild with job_scripts/gb200_setup.sh."
    exit 1
}

# ---- Resume: newest last.ckpt under RUNDIR -------------------------------
# FRESH=1 when reusing a RUNDIR with a different overlay, or the run silently
# continues the previous experiment's optimiser state and step count.
CKPT="$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
if [ "${FRESH:-0}" = "1" ]; then
    echo ">>> FRESH=1: starting from scratch (ignoring ${CKPT:-none})"
    EXTRA=()
elif [ -n "$CKPT" ]; then
    echo ">>> resuming from $CKPT"
    EXTRA=(--ckpt_path "$CKPT")
else
    echo ">>> fresh start (no last.ckpt found)"
    EXTRA=()
fi

# ---- Overlay: LightningCLI applies several --config in order, later wins ---
OVERLAY_ARGS=()
if [ -n "${OVERLAY:-}" ]; then
    [ -f "$OVERLAY" ] || { echo "!!! overlay not found: $OVERLAY"; exit 1; }
    echo ">>> overlay: $OVERLAY"
    OVERLAY_ARGS=(--config "$OVERLAY")
fi

# ---- Stop ten minutes before Slurm does -----------------------------------
# Derived from the allocation rather than written in the overlay, so changing
# -t on the sbatch line is the only edit a different segment length needs.
# Without it the segment is killed mid-epoch and loses whatever has run since
# the last checkpoint - under two minutes at 455 steps an epoch, but for free.
TIME_ARGS=()
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
    SECS=$(( SLURM_JOB_END_TIME - $(date +%s) - 600 ))
    if [ "$SECS" -gt 300 ]; then
        MT=$(printf '00:%02d:%02d:%02d' $((SECS/3600)) $((SECS%3600/60)) $((SECS%60)))
        TIME_ARGS=(--trainer.max_time "$MT")
        echo ">>> max_time $MT (allocation ends $(date -d "@$SLURM_JOB_END_TIME" '+%F %T'))"
    fi
else
    echo ">>> SLURM_JOB_END_TIME unset; no max_time, the segment will be killed at the wall"
fi

# ---- Optional self-chaining ----------------------------------------------
# CHAIN=1 submits the next segment NOW, before training starts, with a
# dependency on this job. Submitting at the start rather than at the end is the
# whole point: a segment that reaches its wall clock is killed and nothing
# after srun ever runs, so a chain built on the last line breaks exactly when
# it is needed. MaxSubmitPU is 2 on both r1 and r2, which is one running plus
# one waiting - so a chained run uses the entire allowance and only ONE
# experiment can chain at a time.
#
# Two stop conditions, and it needs both. The step check ends the chain when
# the run is finished; the hop counter ends it if the step check ever fails to
# parse, which would otherwise resubmit forever.
if [ "${CHAIN:-0}" = "1" ]; then
    CHAIN_LEFT="${CHAIN_LEFT:-12}"
    # train.py's ModelCheckpoint filename ends in -s<step> and last.ckpt is a
    # symlink to it, so the step reads out of the name without loading 305 MB
    # of tensors.
    STEP=0
    if [ "${FRESH:-0}" != "1" ] && [ -n "$CKPT" ]; then
        STEP=$(basename "$(readlink -f "$CKPT")" .ckpt | sed -n 's/.*-s\([0-9]\{1,\}\)$/\1/p')
        STEP="${STEP:-0}"
    fi
    TARGET=$(python - "${OVERLAY:-}" <<'PY'
import sys, yaml
target = 0
for f in ["config.yaml"] + [a for a in sys.argv[1:] if a]:
    with open(f, encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    v = (d.get("trainer") or {}).get("max_steps")
    if v is not None:
        target = v
print(target)
PY
)
    echo ">>> chain: at step $STEP of $TARGET, $CHAIN_LEFT hops left"
    if [ "$CHAIN_LEFT" -gt 0 ] && [ "$STEP" -lt "$TARGET" ]; then
        TLIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o "%l" | tr -d ' ')
        if NEXT=$(sbatch --parsable \
                    --dependency=afterany:"$SLURM_JOB_ID" \
                    --nodes="$SLURM_JOB_NUM_NODES" \
                    --time="$TLIMIT" \
                    --export=ALL,FRESH=0,CHAIN=1,CHAIN_LEFT=$((CHAIN_LEFT - 1)) \
                    job_scripts/train_gb200.sh 2>&1); then
            echo ">>> chained next segment: job $NEXT (-t $TLIMIT)"
        else
            echo "!!! chain submit failed, this segment still runs: $NEXT"
        fi
    else
        echo ">>> chain ends here"
    fi
fi

# Lightning does not read the SBATCH directives; num_nodes and devices have to
# be passed, and srun's task count comes from Slurm rather than a literal, so
# changing --nodes needs no second edit here.
echo "=========================================="
srun -n "$SLURM_NTASKS" python train.py fit \
     --config config.yaml "${OVERLAY_ARGS[@]}" \
     --trainer.num_nodes "$SLURM_JOB_NUM_NODES" \
     --trainer.devices "$SLURM_NTASKS_PER_NODE" \
     "${TIME_ARGS[@]}" \
     --trainer.default_root_dir "$RUNDIR" "${EXTRA[@]}"

echo "=========================================="
date
echo "done"
