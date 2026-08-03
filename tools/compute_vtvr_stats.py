"""
計算 vt / vr / vt10 / vr10 的標準化統計量,並產生新的 stat 檔。

背景:
    資料集 labeled_and_obs_data_with_vt_vr 已內含切向/徑向風,但既有的
    utils/stat_mean.nc、utils/stat_std.nc 沒有這四個變數 ——
    ERA5TCDataset._stat_from_nc 會對變數表逐一取值,缺變數就 KeyError。

本腳本:
    1. 從【訓練年份】隨機抽樣 N 個 *combined.nc
    2. 以單遍累加(sum / sum of squares)算 mean 與 std
       - vt, vr   : 每個氣壓層各一個值(對齊既有 u, v 的 (1,13,1,1) 佈局)
       - vt10,vr10: 單一純量      (對齊既有 u10, v10 的 (1,1,1) 佈局)
    3. 複製原 stat 檔並寫入這四個變數,輸出成 *_vtvr.nc(不動原檔)

用法(在 NCHC 上,cwd = repo 根目錄):
    python tools/compute_vtvr_stats.py
    python tools/compute_vtvr_stats.py --n-sample 1500      # 想更準就抽多一點
"""
import argparse
import glob
import os
import random
import shutil
import sys

import netCDF4
import numpy as np

UPPER = ["vt", "vr"]          # (1, 13, 81, 81) → 每層一個統計值
SURFACE = ["vt10", "vr10"]    # (1, 81, 81)     → 單一純量


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/yungyun0721/TC_dataset/DLDA_data/"
                                      "ERA5_DLDA_data/labeled_and_obs_data_with_vt_vr")
    ap.add_argument("--train-start", type=int, default=2007)
    ap.add_argument("--train-end", type=int, default=2018, help="不含此年(與 config 一致)")
    ap.add_argument("--n-sample", type=int, default=1000)
    ap.add_argument("--src-mean", default="utils/stat_mean.nc")
    ap.add_argument("--src-std", default="utils/stat_std.nc")
    ap.add_argument("--out-mean", default="utils/stat_mean_vtvr.nc")
    ap.add_argument("--out-std", default="utils/stat_std_vtvr.nc")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # ---------- 收集訓練年份的檔案 ----------
    files = []
    for y in range(a.train_start, a.train_end):
        files += glob.glob(os.path.join(a.root, str(y), "*", "*combined.nc"))
    if not files:
        print(f"!! 在 {a.root} 找不到 {a.train_start}-{a.train_end - 1} 的 *combined.nc")
        return 1

    random.seed(a.seed)
    random.shuffle(files)
    sample = files[: a.n_sample]
    print(f"訓練年份 {a.train_start}-{a.train_end - 1}:共 {len(files)} 檔,抽樣 {len(sample)} 檔")

    # ---------- 單遍累加 ----------
    n_lev = None
    acc = {v: {"n": 0, "s": None, "ss": None} for v in UPPER + SURFACE}
    bad = 0

    for i, f in enumerate(sample, 1):
        try:
            with netCDF4.Dataset(f, "r") as d:
                missing = [v for v in UPPER + SURFACE if v not in d.variables]
                if missing:
                    if bad == 0:
                        print(f"!! {os.path.basename(f)} 缺變數 {missing}")
                    bad += 1
                    continue

                for v in UPPER:
                    x = np.asarray(d[v][:], dtype=np.float64)      # (1, L, Y, X)
                    x = x.reshape(x.shape[-3], -1)                  # (L, Y*X)
                    if n_lev is None:
                        n_lev = x.shape[0]
                    if acc[v]["s"] is None:
                        acc[v]["s"] = np.zeros(n_lev)
                        acc[v]["ss"] = np.zeros(n_lev)
                    acc[v]["s"] += x.sum(axis=1)
                    acc[v]["ss"] += (x ** 2).sum(axis=1)
                    acc[v]["n"] += x.shape[1]

                for v in SURFACE:
                    x = np.asarray(d[v][:], dtype=np.float64).ravel()
                    if acc[v]["s"] is None:
                        acc[v]["s"] = 0.0
                        acc[v]["ss"] = 0.0
                    acc[v]["s"] += x.sum()
                    acc[v]["ss"] += (x ** 2).sum()
                    acc[v]["n"] += x.size
        except Exception as e:                                      # noqa: BLE001
            bad += 1
            if bad <= 3:
                print(f"!! 讀取失敗 {os.path.basename(f)}: {e}")

        if i % 200 == 0:
            print(f"  ... {i}/{len(sample)}")

    ok = len(sample) - bad
    if ok == 0:
        print("!! 沒有成功讀到任何檔案")
        return 1
    print(f"成功 {ok} 檔,失敗 {bad} 檔\n")

    # ---------- mean / std ----------
    stats = {}
    for v in UPPER + SURFACE:
        n = acc[v]["n"]
        mean = acc[v]["s"] / n
        var = acc[v]["ss"] / n - mean ** 2
        var = np.maximum(var, 0.0)                                  # 數值保護
        stats[v] = (mean, np.sqrt(var))

    print(f"{'變數':<8} {'mean':>34}  {'std':>34}")
    for v in UPPER:
        m, s = stats[v]
        print(f"{v:<8} {np.array2string(m, precision=3, max_line_width=200)}")
        print(f"{'':<8} std: {np.array2string(s, precision=3, max_line_width=200)}")
    for v in SURFACE:
        m, s = stats[v]
        print(f"{v:<8} mean={m:>10.4f}   std={s:>10.4f}")

    # ---------- 寫出新的 stat 檔 ----------
    for src, out, which in ((a.src_mean, a.out_mean, 0), (a.src_std, a.out_std, 1)):
        if not os.path.exists(src):
            print(f"!! 找不到來源 {src}")
            return 1
        shutil.copyfile(src, out)
        with netCDF4.Dataset(out, "a") as d:
            for v in UPPER:
                if v in d.variables:
                    d.variables[v][:] = stats[v][which].reshape(1, n_lev, 1, 1)
                else:
                    nv = d.createVariable(v, "f8", ("time", "level", "latitude", "longitude"))
                    nv[:] = stats[v][which].reshape(1, n_lev, 1, 1)
            for v in SURFACE:
                if v in d.variables:
                    d.variables[v][:] = np.array(stats[v][which]).reshape(1, 1, 1)
                else:
                    nv = d.createVariable(v, "f8", ("time", "latitude", "longitude"))
                    nv[:] = np.array(stats[v][which]).reshape(1, 1, 1)
        print(f"寫出 {out}")

    # ---------- 驗證 ----------
    print("\n--- 驗證:用 ERA5TCDataset 的讀法確認新檔可用 ---")
    for f in (a.out_mean, a.out_std):
        with netCDF4.Dataset(f) as d:
            got = [v for v in UPPER + SURFACE if v in d.variables]
            print(f"  {f}: 含 {got}  (共 {len(d.variables)} 個變數)")

    print("\n下一步 —— 改 config.yaml:")
    print(f"  data.stat_mean_file: '{a.out_mean}'")
    print(f"  data.stat_std_file:  '{a.out_std}'")
    print("  data.upper_variables / model.upper_vars :  u, v      → vt, vr")
    print("  data.surface_variables / model.surface_vars: u10, v10 → vt10, vr10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
