# LLAT.ty — 極座標版 (polar)

TC 中心極座標 (r, θ) 版的 LLAT.ty 區域颱風預報模式。
源自學姊 (yungyun) 的 `DLAMPty_polar`,整理成扁平結構以便在 NCHC H200 叢集上訓練。

> 完整的診斷分析(bug 清單、架構比較、訓練狀態、論文方向)見另一份
> `POLAR_ANALYSIS_REPORT.md`(在 `couple_FCNV2_LLAT` repo)。

---

## 快速開始(NCHC H200 叢集)

```bash
git clone <你的 repo 網址> LLAT_polar
cd LLAT_polar
mkdir -p job_logs

# 1) 填入你的計畫代號與 email(兩支腳本都要)
sacctmgr show assoc user=$USER format=Account -n | sort -u   # 查代號
vi job_scripts/train_h200.sh      # 改 --account / --mail-user
vi job_scripts/calibrate.sh       # 改 --account

# 2) 校準(4h 的 dev partition,約 15 分鐘)
sbatch job_scripts/calibrate.sh
#   或互動式(回饋更快):
#   salloc -A <account> -p dev -N 1 --gpus-per-node=8 --ntasks-per-node=8 \
#          --cpus-per-task=12 -t 00:30:00
#   bash job_scripts/calibrate.sh

# 3) 用校準結果填 config.yaml 的 max_steps(見下)

# 4) 正式訓練,三段接力(每段 48h)
J1=$(sbatch --parsable job_scripts/train_h200.sh)
J2=$(sbatch --parsable --dependency=afterany:$J1 job_scripts/train_h200.sh)
J3=$(sbatch --parsable --dependency=afterany:$J2 job_scripts/train_h200.sh)
echo "$J1 -> $J2 -> $J3"
```

`train_h200.sh` 會**自動偵測 `last.ckpt` 並續訓**,所以三段用同一支腳本。

---

## ⚠️ 最重要的一件事:`max_steps` 必須設成跑得完的數字

`max_steps` 同時決定 **cosine 學習率的退火曲線**。

前一次訓練的教訓:`max_steps: 1600000` 但實際只跑到 252,160(**15.8%**)
⇒ 學習率只從 4.0e-4 降到 3.8e-4(**僅退火 5%**)
⇒ 模型從未進入「精細微調」階段,val_loss 停在 0.29 看似收斂,其實是還在高溫跳動。

**正確做法**:
```
max_steps = (校準量到的每秒步數) × (總預算秒數) × 0.9
例:1.0 步/秒 × 48h × 3 段 × 3600 × 0.9 ≈ 466,000
```

---

## 校準要看什麼

| 觀察 | 判讀 |
|---|---|
| progress bar 的 `it/s` | 每秒幾步 → 回推 `max_steps` |
| 一個 epoch 的總步數 | 應 ≈ 樣本數 /(`batch_size` × 8)。若接近 樣本數/`batch_size`(沒除以 8)⇒ **DDP 沒生效** |
| `nvidia-smi` 的 GPU util | < 80% ⇒ dataloader 是瓶頸,調高 `n_workers` |

---

## 資料

```
/work/yungyun0721/TC_dataset/DLDA_data/ERA5_DLDA_data/labeled_and_obs_data_with_vt_vr
```
- 2007–2020(14 年)、394 個 TC、19,984 個 `*combined.nc`
- → 約 **19,590** 個 (input, target) 樣本;訓練年份 2007–2017 約 15,400 個
- 網格 81×81、13 層(模式內部再轉成極座標 201×180)
- **已內含 `vt`/`vr`/`vt10`/`vr10`**(切向/徑向風,無 NaN)→ 可直接做 Vt/Vr 消融實驗,不需自行計算

---

## 目前設定

| 項目 | 值 | 備註 |
|---|---|---|
| 網格 | 201 × 180 (r × θ) | r_max=10°、Δr=0.05°、Δθ=2° |
| patch / window | (2,8,6) / (2,10,15) + (2,8,10) | layer1&4 / layer2&3 |
| 參數量 | 24,570,650 | |
| precision | `16-mixed` | 原為 `32`(慢 2 倍) |
| batch_size | 4 /GPU | 8 卡 ⇒ 等效 32 |
| n_workers | 10 | 每 task 上限 12 cores |

---

## 已知問題(訓練不受影響,推論才會踩到)

| # | 位置 | 問題 |
|---|---|---|
| B1 | `DLAMPty_inference.py:319` | `polar_to_latlon(fill_value=0.0)` → 圓外填**物理 0**(應為 NaN);繪圖會出現彩虹環與異常區 |
| B2 | `utils/data_processor.lonlat_uniformizer` | 中心用全陣列平均;若吃到 B1 的 0,TC 中心會偏約 **3400 km**(潛伏未爆彈) |
| B3 | `DLAMPty_inference.py:43` | `r` 從 0 起算 → 中心奇點(最內圈 180 個 θ 採到同一點) |
| B4 | `export_onnx.py` | `input_sample` 仍是 81×81 笛卡兒形狀,對 polar 模型會 shape mismatch |
| B7 | `models/pangu_polar.py` `EarthAttention3D` | `earth_specific_bias` 全零、非 Parameter;每步多一次 CPU→GPU 搬運 |

---

## 測試

```bash
python -m pytest tests/ -v
```
- `test_smoke` / `test_determinism` / `test_backward_grad_coverage` — 會過
- `test_theta_equivariance` — 標為 **xfail 診斷**(等變性非驗收標準,詳見報告)
