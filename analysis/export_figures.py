"""執行 training_analysis.ipynb,把圖匯出成 PNG【並把輸出寫回 notebook】。

用法(cwd = analysis/,環境用 ty-dev):
    python export_figures.py             # 匯出 PNG + 更新 notebook 內嵌輸出
    python export_figures.py --png-only  # 只匯出 PNG,不動 notebook

輸出到 figures/,檔名依 notebook 裡的圖號。重跑會覆蓋。

為什麼要寫回 notebook(2026-08-05 修):
    舊版只寫 figures/*.png,不碰 .ipynb。於是「新資料進來 → 跑一次匯出」之後,
    figures/ 是新的,但在 VS Code 打開 notebook 看到的仍是【上次在 Jupyter
    執行時】存下的舊圖 —— 兩個產物靜默分岔,而人看的是 notebook。
    實際踩過:兩組正式訓練的 curves.csv 到位後圖已重新匯出,但 notebook 裡
    存的仍是「(略過,檔案不存在:prod_*.csv)」與 6 條線的舊圖。
    現在一次執行同時更新兩者,不可能再分岔。

作法:用 Agg 後端依序 exec notebook 的 code cell,把 plt.show() 換成
「存檔 + 收進本 cell 的輸出」,並攔截 stdout。不需要 nbconvert / jupyter。
"""
import base64
import contextlib
import io
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
    "fig05_gradient_norm",      # 梯度範數八組對照 ★
    "fig05b_val_loss_states",   # 損失狀態(全程 / 同段 step / vs LR)
    "fig06_confounder",         # 診斷的混淆變因
    "fig07_epicentre",          # 哪一層先失控 ★
    "fig08_grids",              # 三種網格幾何 ★
    "fig09_precision_matrix",   # 精度 × 學習率矩陣 ★
    "fig10_throughput",         # 80x 吞吐分解
    "fig11_sampling_density",   # 取樣密度 vs 半徑
    "fig12_interpolation",      # 內插與 r=0 奇點 ★
    "fig13_coupling",           # 耦合的空間分工 ★
    "fig14_prod_gradient",      # 正式訓練 A/B:梯度 ★
    "fig15_prod_loss",          # 正式訓練 A/B:損失與泛化落差 ★
]


def main():
    png_only = "--png-only" in sys.argv
    nb_path = "training_analysis.ipynb"
    if not os.path.exists(nb_path):
        sys.exit("找不到 %s —— 請在 analysis/ 目錄下執行" % nb_path)
    os.makedirs(OUTDIR, exist_ok=True)

    saved = []          # 已存檔的 PNG 路徑(跨 cell 累計,決定檔名)
    pending = []        # 本 cell 產生、尚未歸戶的 PNG bytes

    def save_open_figures():
        """plt.show() 的替身:存檔 + 收進 pending,等 cell 跑完再歸戶。"""
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = (NAMES[len(saved)] if len(saved) < len(NAMES)
                    else "fig_extra%02d" % (len(saved) + 1))
            path = os.path.join(OUTDIR, name + ".png")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            saved.append(path)
            pending.append(open(path, "rb").read())
            plt.close(fig)

    plt.show = save_open_figures

    nb = json.load(open(nb_path, encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    env = {}
    for i, c in enumerate(code_cells):
        src = "".join(c["source"])
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, "<cell %d>" % i, "exec"), env)
                save_open_figures()      # 收掉本 cell 沒 show 的圖
        except Exception as e:
            sys.stdout.write(buf.getvalue())
            print("!!! cell %d 失敗: %s: %s" % (i, type(e).__name__, e))
            raise
        text = buf.getvalue()
        sys.stdout.write(text)

        if not png_only:
            # 先圖後文:本 notebook 幾乎都是 plt.show() 之後才 print 摘要。
            outs = [{"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(p).decode()}}
                    for p in pending]
            if text:
                outs.append({"output_type": "stream", "name": "stdout",
                             "text": text.splitlines(keepends=True)})
            c["outputs"] = outs
            c["execution_count"] = i + 1
        pending.clear()

    if not png_only:
        # 全部 cell 都成功才寫回,避免中途失敗留下半套輸出。
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("\n已更新 %s 的內嵌輸出(%.1f MB)"
              % (nb_path, os.path.getsize(nb_path) / 1e6))

    print("\n匯出 %d 張圖到 %s/" % (len(saved), OUTDIR))
    for p in saved:
        print("   %-42s %6.0f KB" % (p, os.path.getsize(p) / 1024))
    if len(saved) != len(NAMES):
        print("\n⚠️ 圖數(%d)與 NAMES(%d)不符 —— 請更新 NAMES 清單。"
              % (len(saved), len(NAMES)))


if __name__ == "__main__":
    main()
