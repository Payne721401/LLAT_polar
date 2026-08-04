"""把 Lightning 的 TensorBoard event 檔轉成小 CSV,方便下載與繪圖(不需 tensorboard)。

用法(cwd = repo 根目錄):
    python tools/events_to_csv.py                          # 自動找最新的 version_*
    python tools/events_to_csv.py lightning_logs/version_233052
    python tools/events_to_csv.py <dir> --all              # 連逐變數指標一起輸出

輸出:
    <version_dir>/curves.csv        主要曲線(val_loss / train_loss_epoch / lr / epoch)
    <version_dir>/per_var.csv       逐變數、逐層的 val_RMSE 與 val_norm_L1(加 --all 才輸出)

CSV 只有幾百 KB,scp 很快(event 檔本身可能上百 MB)。
"""
import collections
import csv
import glob
import os
import struct
import sys

MAIN = ("val_loss", "train_loss_epoch", "train_loss_step", "lr-AdamW", "epoch",
        "gradient_2norm")


def varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def fields(b):
    i, n = 0, len(b)
    while i < n:
        k, i = varint(b, i)
        fn, wt = k >> 3, k & 7
        if wt == 0:
            v, i = varint(b, i); yield fn, wt, v
        elif wt == 1:
            yield fn, wt, b[i:i + 8]; i += 8
        elif wt == 2:
            L, i = varint(b, i); yield fn, wt, b[i:i + L]; i += L
        elif wt == 5:
            yield fn, wt, b[i:i + 4]; i += 4
        else:
            return


def parse(path):
    series = collections.defaultdict(list)
    data = open(path, "rb").read()
    i, n = 0, len(data)
    while i + 12 <= n:
        L = struct.unpack_from("<Q", data, i)[0]; i += 12
        if i + L + 4 > n:
            break
        rec = data[i:i + L]; i += L + 4
        step = summ = wall = None
        for fn, wt, v in fields(rec):
            if fn == 1 and wt == 1:
                wall = struct.unpack("<d", v)[0]
            elif fn == 2 and wt == 0:
                step = v
            elif fn == 5 and wt == 2:
                summ = v
        if summ is None:
            continue
        for fn, wt, v in fields(summ):
            if fn != 1 or wt != 2:
                continue
            tag = val = None
            for f2, w2, v2 in fields(v):
                if f2 == 1 and w2 == 2:
                    try:
                        tag = v2.decode()
                    except Exception:
                        pass
                elif f2 == 2 and w2 == 5:
                    val = struct.unpack("<f", v2)[0]
            if tag and val is not None:
                series[tag].append((step or 0, val, wall))
    return series


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_all = "--all" in sys.argv

    if args:
        vdir = args[0]
    else:
        c = sorted(glob.glob("lightning_logs/version_*"), key=os.path.getmtime)
        if not c:
            sys.exit("找不到 lightning_logs/version_*")
        vdir = c[-1]

    evs = sorted(glob.glob(os.path.join(vdir, "events.out.tfevents.*")))
    if not evs:
        sys.exit(f"{vdir} 裡沒有 event 檔")
    print(f"解析 {evs[0]}  ({os.path.getsize(evs[0])/1e6:.0f} MB) ...")
    S = parse(evs[0])
    print(f"共 {len(S)} 個 tag\n")

    # ---- 主曲線:以 step 為列 ----
    steps = sorted({s for t in MAIN for s, _, _ in S.get(t, [])})
    lut = {t: dict((s, v) for s, v, _ in S.get(t, [])) for t in MAIN}
    t0 = min((w for t in MAIN for _, _, w in S.get(t, []) if w), default=None)
    wall = {}
    for t in MAIN:
        for s, _, w in S.get(t, []):
            if w and s not in wall:
                wall[s] = w

    out = os.path.join(vdir, "curves.csv")
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "hours"] + list(MAIN))
        for s in steps:
            h = (wall[s] - t0) / 3600 if (t0 and s in wall) else ""
            wr.writerow([s, f"{h:.4f}" if h != "" else ""] +
                        [f"{lut[t][s]:.6g}" if s in lut[t] else "" for t in MAIN])
    print(f"寫出 {out}  ({os.path.getsize(out)/1024:.0f} KB, {len(steps)} 列)")

    # ---- 摘要 ----
    print()
    for t in MAIN:
        p = sorted(S.get(t, []))
        if not p:
            continue
        vals = [v for _, v, _ in p]
        b = min(vals); bs = p[vals.index(b)][0]
        print(f"  {t:<18} 首 {vals[0]:>9.5f}  末 {vals[-1]:>9.5f}  最佳 {b:>9.5f} @ step {bs}")

    # ---- 逐變數指標 ----
    if want_all:
        per = [t for t in S if t.startswith(("val_RMSE/", "val_norm_L1/", "grad_2.0_norm/"))]
        out2 = os.path.join(vdir, "per_var.csv")
        with open(out2, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["tag", "step", "value"])
            for t in sorted(per):
                for s, v, _ in sorted(S[t]):
                    wr.writerow([t, s, f"{v:.6g}"])
        print(f"\n寫出 {out2}  ({os.path.getsize(out2)/1024:.0f} KB, {len(per)} 個變數)")

        print("\n最終 val_RMSE 前 15 名(數值大 = 學得差):")
        fin = sorted(((sorted(S[t])[-1][1], t) for t in per if t.startswith("val_RMSE/")),
                     reverse=True)
        for v, t in fin[:15]:
            print(f"    {t:<28} {v:>12.4f}")


if __name__ == "__main__":
    main()
