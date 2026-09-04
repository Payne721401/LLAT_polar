#!/bin/bash
# ============================================================================
# LLAT polar - NCHC GB200 (aarch64) training. Companion to train_h200.sh, which
# stays the script for the H200 cluster; the two are not interchangeable.
#
#   sbatch --export=ALL,SAVE_TOP_K=5,FRESH=1,RUNDIR=/work/$USER/runs/NAME,OVERLAY=experiments/NAME.yaml \
#          job_scripts/train_gb200.sh
#   sbatch --dependency=afterany:<JobID> --export=ALL,SAVE_TOP_K=5,RUNDIR=... job_scripts/train_gb200.sh
#
# Four differences from the H200 script, every one forced by the machine.
#
# 1. Four GPUs per node, not eight. Eight-way DDP is two nodes, and batch_size
#    is per device, so the effective batch is nodes x 4 x batch_size. The H200
#    runs are 8 x 4 = 32; 2 nodes x 4 x 4 reproduces that exactly, and one node
#    needs batch_size 8 to match. An effective batch that does not match is a
#    different experiment, not a faster one - max_steps defines the cosine
#    curve and the per-sample learning rate moves with it.
#
# 2. gb200-r1 allows 24 h against 8gpus' 48 h, so a run needs twice the
#    segments. QOS p_gb200_r1 allows two jobs per user, which is exactly "one
#    running plus one chained" - do not queue a second follow-up, it will be
#    rejected and the chain will silently stop at the first link.
#
# 3. aarch64. The cluster's miniconda3 is an x86-64 build and cannot execute on
#    these nodes at all ("cannot execute binary file"), and PyTorch stopped
#    publishing conda packages after 2.5 while Blackwell needs >= 2.7. So there
#    is no conda here: the environment is a venv built with /usr/bin/python3.11
#    and pip, and the CUDA runtime comes from the wheels rather than a module.
#    --export=ALL still drags the login node's x86 conda in, which is why the
#    environment is stripped below before the venv is sourced.
#
# 4. Checkpoints go to /work. /home is 100 GB with ~45 GB free and one
#    checkpoint is 305 MB, so a default save_top_k of 30 is 9.2 GB per run and
#    a full quota kills the job mid-write with an error that never mentions
#    disks. Set SAVE_TOP_K on the sbatch line.
# ============================================================================
#SBATCH --account=MST115002
#SBATCH --job-name=LLAT_gb200
#SBATCH --partition=gb200-r1
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4           # one task per GPU (DDP)
#SBATCH --cpus-per-task=32            # 136 effective cores / 4 tasks, with headroom
#SBATCH --time=23:55:00               # partition limit is 24 h
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

# ---- Run directory. /work, not /home; see note 4 above. ----
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

# Lightning does not read the SBATCH directives; num_nodes and devices have to
# be passed, and srun's task count comes from Slurm rather than a literal, so
# changing --nodes above needs no second edit here.
echo "=========================================="
srun -n "$SLURM_NTASKS" python train.py fit \
     --config config.yaml "${OVERLAY_ARGS[@]}" \
     --trainer.num_nodes "$SLURM_JOB_NUM_NODES" \
     --trainer.devices "$SLURM_NTASKS_PER_NODE" \
     --trainer.default_root_dir "$RUNDIR" "${EXTRA[@]}"

echo "=========================================="
date
echo "done"
