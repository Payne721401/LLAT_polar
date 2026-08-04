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
#SBATCH --account=MST115002          # ← 必填!用 sacctmgr show assoc user=$USER 查
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
#SBATCH --mail-user=90727sam@gmail.com        # ← 填你的 email

set -eo pipefail          # 注意:不要加 -u,conda 的 activate.d 腳本會引用未定義變數
mkdir -p job_logs

echo "=========================================="
date
echo "Job     : ${SLURM_JOB_ID:-NA}"
echo "Node    : $(hostname)"
echo "WorkDir : $(pwd)"

# ---- 執行目錄:log 與 checkpoint 都寫在這裡下面的 lightning_logs/ ----
# 並行跑多組實驗時每組給不同的 RUNDIR,否則兩個 job 會搶同一個 version 編號:
#     sbatch --export=ALL,RUNDIR=runs/diag_fp32,OVERLAY=... job_scripts/train_h200.sh
RUNDIR="${RUNDIR:-.}"
mkdir -p "$RUNDIR"
echo "RunDir  : $RUNDIR"

# 模組名以本叢集 `ml avail` 為準(與 nano5 不同)
module load miniconda3 || true          # 通常已預設載入
module load gcc/11.5   || true
module load cuda/12.6  || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u                                   # conda 的 activate.d 會引用未定義變數
# sbatch 預設 --export=ALL,會把「送出 job 的登入 shell 已活化的 conda 環境」
# 整包帶進計算節點,和節點上 module load 的 conda 打架:
# conda activate 以為已在 ty 而空轉,PATH 卻已被 module 蓋成 base → import torch 失敗。
# 先把繼承來的環境剝乾淨,再活化一次。
while [ -n "${CONDA_PREFIX:-}" ]; do conda deactivate || break; done
conda activate ty
# 不要重新開啟 -u:第一次訓練時 EXTRA 是空陣列,
# "${EXTRA[@]}" 在 set -u 下展開空陣列會報 unbound variable 而中止

echo "--- GPU ---"
nvidia-smi -L
echo "--- Python ---"
echo "CONDA_PREFIX = ${CONDA_PREFIX:-<空>}"
echo "python       = $(command -v python)"
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| n_gpu',torch.cuda.device_count())" || {
    echo "!!! 環境活化失敗:ty 裡找不到 torch。檢查 conda env list / module load。"
    exit 1
}

# ---- 自動續訓:在 RUNDIR 底下找最新的 last.ckpt ----
# 沿用同一個 RUNDIR 卻換 overlay 時務必加 FRESH=1,否則會誤續上一組的 checkpoint:
#     sbatch --export=ALL,FRESH=1 job_scripts/train_h200.sh
CKPT="$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
if [ "${FRESH:-0}" = "1" ]; then
    echo ">>> FRESH=1:強制全新訓練(忽略 ${CKPT:-無})"
    EXTRA=()
elif [ -n "$CKPT" ]; then
    echo ">>> 從 checkpoint 續訓: $CKPT"
    EXTRA=(--ckpt_path "$CKPT")
else
    echo ">>> 全新訓練(找不到 last.ckpt)"
    EXTRA=()
fi

# ---- 疊加式 config:對照實驗用 overlay 覆蓋部分設定,不必改 config.yaml ----
# LightningCLI 會依序套用多個 --config,後者覆蓋前者。例:
#     sbatch --export=ALL,FRESH=1,OVERLAY=experiments/diag_fp32.yaml job_scripts/train_h200.sh
OVERLAY_ARGS=()
if [ -n "${OVERLAY:-}" ]; then
    [ -f "$OVERLAY" ] || { echo "!!! 找不到 overlay 檔: $OVERLAY"; exit 1; }
    echo ">>> 疊加 overlay: $OVERLAY"
    OVERLAY_ARGS=(--config "$OVERLAY")
fi

echo "=========================================="
srun -n 8 python train.py fit --config config.yaml "${OVERLAY_ARGS[@]}" \
     --trainer.default_root_dir "$RUNDIR" "${EXTRA[@]}"

echo "=========================================="
date
echo "done"
