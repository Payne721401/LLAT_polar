#!/bin/bash
# ============================================================================
# 短跑校準 —— 量出「每步幾秒」「每 epoch 幾步」,用來決定正式跑的 max_steps
#
# 建議用【互動式】跑,回饋最快:
#   salloc -A <account> -p dev -N 1 --gpus-per-node=8 --ntasks-per-node=8 \
#          --cpus-per-task=12 -t 00:30:00
#   然後在配到的節點上執行:  bash job_scripts/calibrate.sh
#
# 也可以 sbatch 這支(partition 已設 dev / 30 分鐘)
# ============================================================================
#SBATCH --account=MST115002
#SBATCH --job-name=LLAT_cal
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=job_logs/cal-%j.out
#SBATCH --error=job_logs/cal-%j.err

set -euo pipefail
mkdir -p job_logs

module load miniconda3/24.11.1 gcc/11.5.0 cuda/12.4 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ty

nvidia-smi -L
python -c "import torch;print('n_gpu',torch.cuda.device_count())"

echo "=========================================================="
echo "校準跑:只跑 60 個 batch,量吞吐"
echo "觀察重點:"
echo "  1) progress bar 的 it/s  → 每步幾秒"
echo "  2) 一個 epoch 的總步數   → 驗證 DDP 是否真的用到 8 卡"
echo "     期望 steps/epoch ≈ 訓練樣本數 / (batch_size × 8)"
echo "     若接近 樣本數/batch_size(未除以 8)⇒ DDP 沒生效!"
echo "  3) 另開終端機跑 nvidia-smi,看 GPU util"
echo "     若 util < 80% ⇒ dataloader 卡住,調高 config 的 n_workers"
echo "=========================================================="

srun -n 8 python train.py fit --config config.yaml \
    --trainer.limit_train_batches=60 \
    --trainer.limit_val_batches=5 \
    --trainer.max_epochs=1 \
    --trainer.max_steps=-1 \
    --trainer.max_time=null \
    --trainer.enable_progress_bar=true \
    --trainer.log_every_n_steps=5

echo "=========================================================="
echo "校準完成。用下式回推 max_steps:"
echo "  max_steps = (每秒步數) × (總預算秒數) × 0.9"
echo "  例:0.8 步/秒 × 48h × 3 段 × 3600 × 0.9 ≈ 373,000"
echo "=========================================================="
