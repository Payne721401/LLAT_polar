#!/bin/bash
# ============================================================================
# One-shot GB200 (aarch64) environment build plus diagnostics.
#
#   sbatch job_scripts/gb200_setup.sh
#   less job_logs/gb200_setup-<JobID>.out
#
# Run this once before the first training job on GB200, and again whenever the
# venv needs rebuilding. It is a batch job rather than an interactive session
# on purpose: gb200-dev queues behind r1 and r2 for the same physical nodes, so
# waiting for it with an ssh connection open is a good way to lose the work.
#
# Deliberately does NOT set -e. Every section must run so that one log answers
# the whole question, even when an early section fails.
#
# Why a venv and not conda: the cluster's miniconda3 is an x86-64 build and
# cannot execute on these nodes, and PyTorch stopped publishing conda packages
# after 2.5 while Blackwell (sm_100) needs >= 2.7. Nothing here compiles - every
# dependency has an aarch64 wheel, and the CUDA runtime ships inside them.
#
# Only the training path's dependencies are installed. onnxruntime, cartopy,
# xESMF and cdo are in env_building/conda_env.yaml and are the awkward ones on
# aarch64; train.py, models/ and utils/data_modules.py import none of them.
# ============================================================================
#SBATCH --account=MST115002
#SBATCH --job-name=gb200_setup
#SBATCH --partition=gb200-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=job_logs/gb200_setup-%j.out
#SBATCH --error=job_logs/gb200_setup-%j.err

VENV=$HOME/venv-aarch64
DATA=/work/yungyun0721/TC_dataset/DLDA_data/ERA5_DLDA_data/labeled_and_obs_data_with_vt_vr

sec () { echo; echo "==================== $* ===================="; }

sec 1 NODE
date; hostname; uname -m; nproc
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

sec 2 MODULES
module purge 2>/dev/null
module avail 2>&1 | head -60
command -v apptainer singularity

sec 3 DATA
ls -d "$DATA" && ls "$DATA" | wc -l
F=$(find "$DATA" -name '*combined.nc' 2>/dev/null | head -1)
echo "sample: $F"; [ -n "$F" ] && ls -lh "$F"
find "$DATA" -name '*combined.nc' 2>/dev/null | wc -l
df -h "$DATA" "$HOME"

sec 4 PYTHON
# The x86 conda inherited through --export=ALL cannot run here; strip it before
# anything looks up `python`.
unset PYTHONPATH PYTHONHOME
PYBIN=""
for c in /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 /usr/bin/python3; do
    [ -x "$c" ] || continue
    "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null || continue
    PYBIN="$c"; break
done
echo "PYBIN = ${PYBIN:-NONE}"
[ -z "$PYBIN" ] && { echo "!!! no python >=3.10 on this node (match statements need it)"; exit 1; }
"$PYBIN" -V

sec 5 NETWORK
"$PYBIN" - <<'PY'
import urllib.request
try:
    urllib.request.urlopen("https://pypi.org/simple/", timeout=15)
    print("PyPI reachable from compute node")
except Exception as e:
    print("PyPI NOT reachable:", e)
PY

sec 6 VENV
"$PYBIN" -m venv "$VENV" && . "$VENV/bin/activate" || exit 1
python -V; which python
pip install -U pip wheel setuptools

sec 7 TORCH
# 2.11.0 is the first release whose default aarch64 PyPI wheel is CUDA-enabled;
# before it, pip silently installed the CPU build and cuda.is_available() was
# False. Note that section 8 may upgrade this - see the lock file below.
pip install "torch==2.11.0"
python - <<'PY'
import platform, torch
print("machine       ", platform.machine())
print("torch         ", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("capability    ", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a")
PY

sec 8 DEPS
pip install lightning "jsonargparse[signatures]>=4.27.7" timm tensorboard \
            numpy xarray netCDF4 xarray-regrid metpy pysolar matplotlib bottleneck
pip list | grep -Ei "torch|lightning|timm|xarray|netcdf|metpy|pysolar|jsonargparse"

# Installing lightning resolves torch again and can move it off the pin above,
# so record what was actually installed. Rebuilding from this file is what makes
# a later run comparable to an earlier one.
pip freeze > "$HOME/LLAT_polar/env_building/requirements-aarch64.lock.txt"
echo "locked: $(python -c 'import torch; print(torch.__version__)')"

sec 9 IMPORT CHAIN
# Opens utils/land.nc at import time and pulls metpy, pysolar and xarray_regrid,
# so this one line covers the whole data-side dependency chain.
cd "$HOME/LLAT_polar" || exit 1
python -c "import utils.data_modules, models.lightning_modules; print('import ok')"

sec 10 SMOKE TRAIN
mkdir -p "$HOME/scratch"
python train.py fit --config config.yaml \
    --trainer.devices=1 --trainer.num_nodes=1 \
    --trainer.limit_train_batches=20 --trainer.limit_val_batches=2 \
    --trainer.max_epochs=1 --trainer.max_steps=-1 --trainer.max_time=null \
    --trainer.enable_progress_bar=true --trainer.log_every_n_steps=5 \
    --trainer.default_root_dir="$HOME/scratch/gb200_smoke"

sec DONE
date
