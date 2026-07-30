#!/bin/bash
# ============================================================================
# LLAT polar — NCHC H200 叢集訓練腳本(1 節點 × 8 GPU)
#
# 用法:
#   第一次:      sbatch job_scripts/train_h200.sh
#   接力(自動): sbatch --dependency=afterany:<前一個JobID> job_scripts/train_h200.sh
#
#   本腳本會【自動偵測】既有的 last.ckpt 並續訓,所以第一段與後續段可用同一支。
#
# 注意:
#   - partition 8gpus 上限 48 小時 ⇒ 超過需分段接力
#   - 系統限制:每張 GPU 最多 12 CPU cores / 200 GB RAM
#     8 GPU × 12 = 96 cores(節點 112),8 × 200 = 1600 GB(節點 1900 GB)✓
# ============================================================================
#SBATCH --account=REPLACE_ME          # ← 必填!用 sacctmgr show assoc user=$USER 查
#SBATCH --job-name=LLAT_polar
#SBATCH --partition=8gpus             # 48h;校準請改 dev(4h)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8           # 每張 GPU 一個 task(DDP)
#SBATCH --cpus-per-task=12            # 系統上限
#SBATCH --time=2-00:00:00             # 48 小時
#SBATCH --output=job_logs/job-%j.out
#SBATCH --error=job_logs/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REPLACE_ME        # ← 填你的 email

set -euo pipefail
mkdir -p job_logs

echo "=========================================="
date
echo "Job     : ${SLURM_JOB_ID:-NA}"
echo "Node    : $(hostname)"
echo "WorkDir : $(pwd)"

module load miniconda3/24.11.1
module load gcc/11.5.0
module load cuda/12.4
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ty

echo "--- GPU ---"
nvidia-smi -L
echo "--- Python ---"
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| n_gpu',torch.cuda.device_count())"

# ---- 自動續訓:找最新的 last.ckpt ----
CKPT="$(ls -t lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
if [ -n "$CKPT" ]; then
    echo ">>> 從 checkpoint 續訓: $CKPT"
    EXTRA=(--ckpt_path "$CKPT")
else
    echo ">>> 全新訓練(找不到 last.ckpt)"
    EXTRA=()
fi

echo "=========================================="
srun -n 8 python train.py fit --config config.yaml "${EXTRA[@]}"

echo "=========================================="
date
echo "done"
