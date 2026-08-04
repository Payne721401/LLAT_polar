"""把 training_analysis.ipynb 的圖全部匯出成 PNG,供投影片/報告引用。

用法(cwd = analysis/,環境用 ty-dev):
    python export_figures.py

輸出到 figures/,檔名依 notebook 裡的圖號。重跑會覆蓋,所以改完
notebook 只要再跑一次,報告裡的圖就同步更新。

作法:用 Agg 後端依序 exec notebook 的 code cell,並把 plt.show()
換成「存檔後關閉」。不需要 nbconvert / jupyter。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "figures"
DPI = 160

# 圖依【產生順序】命名。notebook 新增圖時在這裡補一行即可。
NAMES = [
    "fig01_loss_curves",        # 損失曲線(mine vs baseline)
    "fig02_lr_annealing",       # 學習率退火對照 ★
    "fig03_lr_vs_stability",    # LR 與 spike 的關係
    "fig04_per_variable_rmse",  # 逐變數 val_RMSE
    "fig05_gradient_norm",      # 梯度範數五組對照 ★
    "fig05b_val_loss_states",   # 損失狀態(全程 / 同段 step / vs LR)
    "fig06_confounder",         # 診斷的混淆變因
    "fig07_epicentre",          # 哪一層先失控 ★
    "fig08_grids",              # 三種網格幾何 ★
    "fig09_precision_matrix",   # 精度 × 學習率矩陣 ★
    "fig10_throughput",         # 80x 吞吐分解
    "fig11_sampling_density",   # 取樣密度 vs 半徑
    "fig12_interpolation",      # 內插與 r=0 奇點 ★
    "fig13_coupling",           # 耦合的空間分工 ★
]


def main():
    nb_path = "training_analysis.ipynb"
    if not os.path.exists(nb_path):
        sys.exit("找不到 %s —— 請在 analysis/ 目錄下執行" % nb_path)
    os.makedirs(OUTDIR, exist_ok=True)

    saved = []

    def save_open_figures():
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = (NAMES[len(saved)] if len(saved) < len(NAMES)
                    else "fig_extra%02d" % (len(saved) + 1))
            path = os.path.join(OUTDIR, name + ".png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            saved.append(path)
            plt.close(fig)

    plt.show = save_open_figures            # 每個 cell 的 plt.show() 改成存檔

    nb = json.load(open(nb_path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    env = {}
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        try:
            exec(compile(src, "<cell %d>" % i, "exec"), env)
        except Exception as e:
            print("!!! cell %d 失敗: %s: %s" % (i, type(e).__name__, e))
            raise

    save_open_figures()                     # 保險:收掉沒被 show 的圖

    print("\n匯出 %d 張圖到 %s/" % (len(saved), OUTDIR))
    for p in saved:
        print("   %-42s %6.0f KB" % (p, os.path.getsize(p) / 1024))
    if len(saved) != len(NAMES):
        print("\n⚠️ 圖數(%d)與 NAMES(%d)不符 —— 請更新 NAMES 清單。"
              % (len(saved), len(NAMES)))


if __name__ == "__main__":
    main()
