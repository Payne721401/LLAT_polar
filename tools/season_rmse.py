"""Per-variable RMSE and bias against lead time, the way a model paper reports it.

season_stats measures the track and season_intensity the storm's own MSLP and
wind. Neither says whether the model gets the ATMOSPHERE right, which is what a
reader coming from WeatherBench2 or the Pangu and GraphCast papers expects: one
row per variable, one column per lead, scored against the analysis.

It is also a comparison this project keeps needing and cannot make. The
validation loss is one number over 98 channels, so when it moves from 0.24 to
0.20 there is no way to see which fields improved. This separates them.

Scope, stated because it is not WeatherBench2. WB2 scores global fields on a
latitude-weighted grid and adds ACC, spectra and ensemble scores. Here the
domain is 20 x 20 degrees around a moving storm, so:

  - No latitude weighting. Over twenty degrees the cos(lat) factor varies by a
    few percent, and the domain is not a latitude band anyway.
  - Scored on the intersection of the runs' valid cells at each lead. Cells
    where any run is NaN - the polar disc corners - are dropped from all of
    them, because a run graded on an easier subset is not being compared.
  - No ACC. Anomaly correlation needs a climatology for the field being scored,
    and a TC-centred moving domain has no published one. Persistence is the
    baseline that applies here, and persistence_baseline.py gives it.

Usage
-----
    python tools/season_rmse.py --era5-root /wk2/yungyun/FCNV2_TC \\
        --runs "cartesian=/wk2/yungyun/FCNV2_TC" \\
        --runs "r80_420k=/home/payne/LLAT_polar_runs_r80long_full" \\
        --limit 40 --csv analysis/season_rmse.csv
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_here, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("season_stats")
pf = ss.pf


def fields(f, sfc_names, upper_pairs):
    """The requested channels as plain arrays, skipping any the run lacks."""
    out = {}
    for v in sfc_names:
        try:
            out[v] = np.asarray(f.s(v), dtype=float)
        except (ValueError, KeyError):
            pass
    for v, lev in upper_pairs:
        try:
            out[v + str(lev)] = np.asarray(f.u(v, lev), dtype=float)
        except (ValueError, KeyError):
            pass
    return out


def main(args):
    era5_root = os.path.expanduser(args.era5_root)
    sfc_names = [v.strip() for v in args.vars.split(",") if v.strip()]
    upper_pairs = []
    for tok in (args.upper or "").split(","):
        tok = tok.strip()
        if ":" in tok:
            v, lev = tok.split(":")
            upper_pairs.append((v.strip(), int(lev)))

    runs = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs.append((label or os.path.basename(spec.rstrip("/")), path or spec))

    found = {}
    for label, path in runs:
        found[label] = {(tc, i): p for tc, i, p in
                        ss.find_starts(path, args.version, args.mode)}
        print("  " + label + ": " + str(len(found[label])) + " cases")
    common = sorted(set.intersection(*(set(v) for v in found.values())))
    print("comparing on the " + str(len(common)) +
          " initial times every run has")

    # acc[label][var][lead] = [sum sq error, sum error, count]
    acc = {l: {} for l, _ in runs}
    leads = list(range(0, int(args.max_lead) + 1, args.every))

    for n, (tc, init_s) in enumerate(common):
        if args.limit and n >= args.limit:
            break
        init = datetime.datetime.strptime(init_s, "%Y%m%d%H")
        for h in leads:
            valid = init + datetime.timedelta(hours=h)
            got, meta = {}, None
            for label, _ in runs:
                try:
                    meta = pf.read_meta(found[label][(tc, init_s)])
                    got[label] = pf.load_run(found[label][(tc, init_s)], h, meta)
                except (FileNotFoundError, OSError, KeyError):
                    got = None
                    break
            if not got:
                continue
            try:
                n_grid = next(iter(got.values())).lon.shape[0]
                truth = pf.load_era5(
                    os.path.join(era5_root, tc, "ERA5", "for_DLAMPty"),
                    tc, valid, n_grid, meta)
            except (FileNotFoundError, OSError, KeyError):
                continue

            t = fields(truth, sfc_names, upper_pairs)
            fs = {l: fields(g, sfc_names, upper_pairs) for l, g in got.items()}
            for var in t:
                if not all(var in fs[l] for l in fs):
                    continue
                # One mask for every run: a forecast whose disc leaves corners
                # NaN would otherwise be graded on a smaller, easier region.
                m = np.isfinite(t[var])
                for l in fs:
                    m = m & np.isfinite(fs[l][var])
                if not m.any():
                    continue
                for l in fs:
                    d = fs[l][var][m] - t[var][m]
                    s = acc[l].setdefault(var, {}).setdefault(h, [0.0, 0.0, 0])
                    s[0] += float(np.sum(d * d))
                    s[1] += float(np.sum(d))
                    s[2] += int(d.size)
        if args.print_every and (n + 1) % args.print_every == 0:
            print("    " + str(n + 1) + "/" + str(len(common)), flush=True)

    order = sfc_names + [v + str(lev) for v, lev in upper_pairs]
    order = [v for v in order if any(v in acc[l] for l, _ in runs)]

    for kind in ("RMSE", "bias"):
        print("\n===== " + kind + " against ERA5 =====")
        head = "{:<10}{:<14}".format("variable", "run")
        print(head + "".join("{:>10}h".format(h) for h in leads))
        print("-" * (24 + 11 * len(leads)))
        for var in order:
            for label, _ in runs:
                cells = []
                for h in leads:
                    s = acc[label].get(var, {}).get(h)
                    if not s or s[2] == 0:
                        cells.append("{:>11}".format("-"))
                        continue
                    v = (s[0] / s[2]) ** 0.5 if kind == "RMSE" else s[1] / s[2]
                    cells.append("{:>11.4g}".format(v))
                print("{:<10}{:<14}".format(var, label) + "".join(cells))
            print()

    draw(acc, order, runs, leads, args)

    if args.csv:
        out = os.path.expanduser(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("run,variable,lead_h,rmse,bias,n\n")
            for label, _ in runs:
                for var in order:
                    for h in leads:
                        s = acc[label].get(var, {}).get(h)
                        if not s or s[2] == 0:
                            continue
                        fh.write("{},{},{},{:.6g},{:.6g},{}\n".format(
                            label, var, h, (s[0] / s[2]) ** 0.5,
                            s[1] / s[2], s[2]))
        print("wrote " + out)


UNITS = {"msl": "Pa", "u10": "m/s", "v10": "m/s", "t2m": "K",
         "tcwv": "kg/m2", "tp": "m", "sp": "Pa", "d2m": "K"}


def unit_of(var):
    if var in UNITS:
        return UNITS[var]
    return {"z": "m2/s2", "t": "K", "u": "m/s", "v": "m/s",
            "q": "kg/kg", "w": "Pa/s"}.get(var[0], "")


def draw(acc, order, runs, leads, args):
    """A grid of RMSE curves, plus one bar panel of the skill difference.

    Two things are wanted from this figure and they need different shapes. The
    per-variable curves answer "how big is the error"; the bars answer "which
    model is better and by how much", which a reader cannot get from eleven
    pairs of overlapping lines. The bars are a percentage, because the variables
    span Pa and kg/kg and an absolute difference is not comparable across them.
    """
    if not order:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(label, var, kind):
        out = []
        for h in leads:
            s = acc[label].get(var, {}).get(h)
            if not s or s[2] == 0:
                out.append(np.nan)
            elif kind == "rmse":
                out.append((s[0] / s[2]) ** 0.5)
            else:
                out.append(s[1] / s[2])
        return np.array(out, dtype=float)

    n = len(order)
    ncol = min(4, n)
    nrow = (n + ncol - 1) // ncol + 1
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow),
                           squeeze=False)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, var in enumerate(order):
        a = ax[i // ncol][i % ncol]
        for (label, _), c in zip(runs, colours):
            a.plot(leads, series(label, var, "rmse"), "-o", ms=3, color=c,
                   label=label)
            if args.show_bias:
                a.plot(leads, np.abs(series(label, var, "bias")), "--",
                       lw=1.0, color=c)
        a.set_title(var + "  [" + unit_of(var) + "]", fontsize=10)
        a.set_xlabel("forecast hour")
        a.set_ylabel("RMSE")
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=8)

    for j in range(n, nrow * ncol):
        r, c = j // ncol, j % ncol
        if r < nrow:
            ax[r][c].axis("off")

    # Bottom row, spanning: the comparison itself. Percentages, because Pa and
    # kg/kg cannot share an axis and the question is relative skill anyway.
    if len(runs) >= 2:
        for c in range(ncol):
            ax[nrow - 1][c].remove()
        bar = fig.add_subplot(nrow, 1, nrow)
        base = runs[0][0]
        width = 0.8 / max(1, len(leads))
        x = np.arange(len(order))
        for k, h in enumerate(leads):
            if h == 0:
                continue
            vals = []
            for var in order:
                b = series(base, var, "rmse")[k]
                o = series(runs[1][0], var, "rmse")[k]
                vals.append(100.0 * (o - b) / b if b and np.isfinite(b) else np.nan)
            bar.bar(x + (k - len(leads) / 2) * width, vals, width,
                    label="+" + str(h) + " h")
        bar.axhline(0, color="0.3", lw=1)
        bar.set_xticks(x)
        bar.set_xticklabels(order, rotation=30, ha="right")
        bar.set_ylabel("RMSE difference [%]")
        bar.set_title(runs[1][0] + " against " + base +
                      " — above zero is worse")
        bar.grid(alpha=0.3, axis="y")
        bar.legend(fontsize=8, ncol=len(leads))

    out = os.path.expanduser(args.out or os.path.join(
        "analysis", "figures", "season",
        "rmse_" + "_".join(l.replace("/", "-") for l, _ in runs) + ".png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("\nwrote " + out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True,
                   metavar="[NAME=]PATH")
    p.add_argument("--era5-root", required=True)
    p.add_argument("--vars", default="msl,u10,v10,t2m,tcwv,tp",
                   help="surface channels, comma separated")
    p.add_argument("--upper", default="z:500,t:850,u:850,v:850,q:700",
                   help="upper-air channels as var:level, comma separated")
    p.add_argument("--version", default=None)
    p.add_argument("--mode", default="one-way",
                   choices=("one-way", "two-way", "standalone"))
    p.add_argument("--every", type=int, default=24)
    p.add_argument("--max-lead", type=float, default=120)
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many cases, for a quick look first")
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--show-bias", action="store_true",
                   help="overlay |bias| as a dashed line on each panel")
    p.add_argument("--out", default=None,
                   help="figure path; defaults to analysis/figures/season/"
                        "rmse_<labels>.png")
    p.add_argument("--csv", default=None)
    main(p.parse_args())
