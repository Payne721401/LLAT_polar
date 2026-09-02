"""How big is the thing a residual connection asks the network to predict?

probe_residual_all lost to probe_base on almost every surface variable while
starting far ahead - msl 171.89 against 480.08 at step 453, level by 20,000.
That shape, a good initialisation with a worse asymptote, is what a network
does when the correction it has to learn is small compared with its own output
scale: the identity map is nearly right, so it sits there and never moves.

That is a claim about a number, and the number is computable from the data
alone, with no model and no GPU:

    ratio = sigma(field at t+dt  -  field at t)  /  sigma(field)

Both are in the same units, so the ratio survives standardisation unchanged -
after (x - mu) / sigma the field has unit variance by construction and the
residual target has variance equal to this ratio. So the ratio IS the scale of
what the network must output, in the units it actually works in.

    ratio near 1     the field changes as much as it varies. The residual has
                     no head start and nothing to explain.
    ratio near 0.3   a normal weather-scale step. Residual learning is easy
                     here and this is where it is usually applied.
    ratio near 0.05  the network must emit a number twenty times smaller than
                     its natural scale, from weights initialised for order one.
                     Any output noise swamps the signal, and the identity map
                     is a local minimum the optimiser has no gradient to leave.

**The step size is read from the filenames, not assumed.** The training data is
3-hourly; the inference boxes under FCNV2_TC are 6-hourly. A ratio measured on
one does not transfer to the other, and the header says which was used.

Statistics are pooled over cells and pairs per channel: the interesting
quantity is how big the change is against how much the field varies, and both
are properties of the channel rather than of one map.

Usage
-----
    # on the lab host, against the inference boxes (6-hourly)
    python tools/residual_scale.py --root /wk2/yungyun/FCNV2_TC \\
        --era5-sub ERA5/for_DLAMPty --limit 40

    # on the cluster, against the training data itself (3-hourly)
    python tools/residual_scale.py --root /path/to/ERA5_for_TC/2019 --limit 40
"""
import argparse
import glob
import os
import re

import numpy as np


def pairs_in(case_dir, name_pattern="*combined.nc"):
    """Adjacent files in one case, with the gap in hours read from the names."""
    files = sorted(glob.glob(os.path.join(case_dir, name_pattern)))
    out = []
    for a, b in zip(files, files[1:]):
        ta, tb = (re.search(r"_(\d{10})_", os.path.basename(f)) for f in (a, b))
        if not ta or not tb:
            continue
        import datetime
        da = datetime.datetime.strptime(ta.group(1), "%Y%m%d%H")
        db = datetime.datetime.strptime(tb.group(1), "%Y%m%d%H")
        out.append((a, b, (db - da).total_seconds() / 3600.0))
    return out


def main(args):
    import xarray as xr

    root = os.path.expanduser(args.root)
    cases = sorted(glob.glob(os.path.join(root, "*")))
    if args.era5_sub:
        cases = [os.path.join(c, args.era5_sub) for c in cases]
    cases = [c for c in cases if os.path.isdir(c)]
    if not cases:
        raise SystemExit(f"no case directories under {root}")

    # sum, sum of squares, count - for the field and for the change
    acc = {}
    gaps, n_pairs = {}, 0
    for c in cases:
        if args.limit and n_pairs >= args.limit:
            break
        for a, b, gap in pairs_in(c):
            if args.limit and n_pairs >= args.limit:
                break
            if args.gap and abs(gap - args.gap) > 0.01:
                continue
            try:
                with xr.open_dataset(a) as da, xr.open_dataset(b) as db:
                    for v in args.vars.split(","):
                        v = v.strip()
                        if not v or v not in da or v not in db:
                            continue
                        x = np.squeeze(da[v].values).astype(float)
                        y = np.squeeze(db[v].values).astype(float)
                        if x.shape != y.shape:
                            continue
                        d = y - x
                        m = np.isfinite(x) & np.isfinite(d)
                        if not m.any():
                            continue
                        s = acc.setdefault(v, [0.0, 0.0, 0.0, 0.0, 0])
                        s[0] += float(np.sum(x[m]))
                        s[1] += float(np.sum(x[m] ** 2))
                        s[2] += float(np.sum(d[m]))
                        s[3] += float(np.sum(d[m] ** 2))
                        s[4] += int(m.sum())
            except (FileNotFoundError, OSError, KeyError):
                continue
            gaps[gap] = gaps.get(gap, 0) + 1
            n_pairs += 1
            if args.print_every and n_pairs % args.print_every == 0:
                print(f"    {n_pairs} pairs", flush=True)

    if not acc:
        raise SystemExit("no usable pairs - check --root, --era5-sub and --vars")

    step = max(gaps, key=gaps.get)
    print(f"\n{n_pairs} pairs, step {step:g} h "
          f"({gaps[step]} of them; other gaps {sorted(k for k in gaps if k != step)})")
    print(f"\n{'variable':<14}{'sigma(field)':>16}{'sigma(change)':>16}"
          f"{'ratio':>10}   verdict")
    print("-" * 74)
    rows = []
    for v in args.vars.split(","):
        v = v.strip()
        if v not in acc:
            continue
        sx, sxx, sd, sdd, n = acc[v]
        sig_x = (sxx / n - (sx / n) ** 2) ** 0.5
        sig_d = (sdd / n - (sd / n) ** 2) ** 0.5
        r = sig_d / sig_x if sig_x else float("nan")
        rows.append((v, r))
        verdict = ("residual is nearly free" if r > 0.6 else
                   "residual is easy" if r > 0.25 else
                   "residual is awkward" if r > 0.1 else
                   "IDENTITY IS A TRAP")
        print(f"{v:<14}{sig_x:>16.5g}{sig_d:>16.5g}{r:>10.3f}   {verdict}")

    hard = [v for v, r in rows if r <= 0.1]
    print(f"\nThe ratio is the standard deviation of the residual target after")
    print(f"standardisation, because standardising divides both by the same")
    print(f"number. A network whose weights are initialised for order-one")
    print(f"outputs has to emit {1/max(min(r for _, r in rows), 1e-9):.0f}x less")
    print(f"than that for the tightest channel here.")
    if hard:
        print(f"\nBelow 0.1: {', '.join(hard)}")
        print("For these the identity map is so nearly correct that there is")
        print("almost no gradient pointing away from it, which is what a run")
        print("that starts far ahead and finishes behind looks like.")
    else:
        print("\nNothing below 0.1, so the residual target is not too small to")
        print("learn and the residual's failure needs a different explanation.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="directory of case directories")
    p.add_argument("--era5-sub", default="ERA5/for_DLAMPty",
                   help="path inside each case to the combined.nc files; pass "
                        "an empty string when --root already holds them")
    p.add_argument("--vars",
                   default="msl,u10,v10,t2m,d2m,sp,tcwv,tp,mtnlwrf,sst,"
                           "landmask,hgt",
                   help="surface channels to measure. Names as they appear in "
                        "the file, which are ERA5's rather than the model's")
    p.add_argument("--gap", type=float, default=0.0,
                   help="only use pairs this many hours apart; 0 uses whatever "
                        "adjacent files give, and the header reports it")
    p.add_argument("--limit", type=int, default=60,
                   help="stop after this many pairs. The statistic converges "
                        "quickly - it is pooled over every cell of every map")
    p.add_argument("--print-every", type=int, default=20)
    main(p.parse_args())
