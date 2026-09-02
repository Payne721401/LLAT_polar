"""Per-variable RMSE split by distance from the storm centre.

The polar formulation's entire claim is inner-core resolution: rings near the
centre get the same number of azimuthal points as rings near the rim, so the
eyewall is sampled far more finely than a Cartesian grid can manage at the same
cost. Every RMSE in this project so far has been a domain average, which mixes
that region with the nine tenths of the box where the claim does not apply -
and a domain average is exactly the statistic that would hide a real inner-core
win under a larger outer-region loss, or the reverse.

So: the same fields, the same pairing, the same clipping, split into rings.

**The rings.** Defaults in km, and they are meteorological rather than round
numbers:

    0-100     eyewall and inner core. RMW here is 0.3-0.5 deg, i.e. 33-55 km,
              so this ring is the one the polar grid was built for.
    100-300   inner rainbands and the primary circulation.
    300-600   outer circulation.
    600-1110  environment. 1110 km is 10 deg, the edge of the polar disc -
              beyond it the polar runs are NaN and there is nothing to compare.

**Distance is in kilometres, not grid cells.** The meridional spacing is a
constant 27.8 km but the zonal spacing is 0.25 deg x cos(lat), which is 8 %
shorter at 35 N than at the equator. Rings measured in cells would therefore be
different physical sizes for a recurving storm than for one in the deep
tropics, and the recurving ones are where the errors live.

**Scored on the intersection of finite cells**, as season_rmse does: a polar
run is NaN outside its disc and a ring that is complete for one run and partial
for another is not a comparison.

Usage
-----
    python tools/season_radial_rmse.py --era5-root /wk2/yungyun/FCNV2_TC \\
      --ibtracs /home/payne/ibtracs/ibtracs.WP.list.v04r01.csv \\
      --clip-to-best-track \\
      --runs "cart_1way=/wk2/yungyun/FCNV2_TC@one_way_couple_model" \\
      --runs "polar_1way=/home/payne/LLAT_polar_runs_r80long_full@one-way" \\
      --max-lead 192 --every 24 --lead 120
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
sr = _load("season_rmse")
pf = ss.pf
DEG_KM = 111.32


def ring_index(field, edges):
    """Which ring each cell falls in, or -1 outside the last edge.

    Distance from the array centre in km, with the cos(lat) factor on
    longitude. The centre is the storm by construction - both the forecast and
    the ERA5 box are storm-following - so no search is needed.
    """
    lon = np.asarray(field.lon, dtype=float)
    lat = np.asarray(field.lat, dtype=float)
    n = lon.shape[0]
    c = (n - 1) // 2
    dx = (lon - lon[c, c]) * DEG_KM * np.cos(np.deg2rad(lat))
    dy = (lat - lat[c, c]) * DEG_KM
    r = np.hypot(dx, dy)
    idx = np.full(r.shape, -1, dtype=int)
    for k in range(len(edges) - 1):
        idx[(r >= edges[k]) & (r < edges[k + 1])] = k
    return idx


def main(args):
    edges = [float(x) for x in args.edges.split(",")]
    names = [f"{edges[k]:.0f}-{edges[k+1]:.0f}" for k in range(len(edges) - 1)]
    sfc_names = [v.strip() for v in args.vars.split(",") if v.strip()]
    upper_pairs = []
    for spec in (args.upper or "").split(","):
        if spec.strip():
            v, _, lev = spec.partition(":")
            upper_pairs.append((v.strip(), int(lev)))

    runs = [ss.parse_run(s, args.mode) for s in args.runs]
    if (args.clip_to_best_track or args.min_vmax) and not args.ibtracs:
        raise SystemExit("--clip-to-best-track and --min-vmax need --ibtracs")
    if args.ibtracs:
        print(f"IBTrACS: {len(ss.ibt.load(args.ibtracs))} storms", flush=True)

    found = {}
    for label, root, mode, subdir in runs:
        found[label] = {(tc, i): p for tc, i, p in
                        ss.find_starts(root, args.version, mode, subdir)}
        print(f"  {label}: {len(found[label])} cases")
    common = sorted(set.intersection(*(set(v) for v in found.values())))
    print(f"comparing on the {len(common)} initial times every run has",
          flush=True)

    era5_root = os.path.expanduser(args.era5_root)
    leads = list(range(0, int(args.max_lead) + 1, args.every))
    # acc[label][var][lead][ring] = [sum of squares, sum, count]
    acc = {l: {} for l, *_ in runs}

    for n, (tc, init_s) in enumerate(common):
        if args.limit and n >= args.limit:
            break
        init = datetime.datetime.strptime(init_s, "%Y%m%d%H")
        clip = None
        if args.clip_to_best_track or args.min_vmax:
            sid = ss.storm_sid(args.ibtracs, tc, init_s,
                               found[runs[0][0]][(tc, init_s)])
            clip = (ss.storm_times(args.ibtracs, sid, args.min_vmax)
                    if sid else set())
        for h in leads:
            valid = init + datetime.timedelta(hours=h)
            if clip is not None and valid not in clip:
                continue
            got, meta = {}, None
            try:
                for label, *_ in runs:
                    meta = pf.read_meta(found[label][(tc, init_s)])
                    got[label] = pf.load_run(found[label][(tc, init_s)], h, meta)
                n_grid = next(iter(got.values())).lon.shape[0]
                truth = pf.load_era5(
                    os.path.join(era5_root, tc, "ERA5", "for_DLAMPty"),
                    tc, valid, n_grid, meta)
            except (FileNotFoundError, OSError, KeyError):
                continue

            rings = ring_index(truth, edges)
            t = sr.fields(truth, sfc_names, upper_pairs)
            fs = {l: sr.fields(g, sfc_names, upper_pairs) for l, g in got.items()}
            for var in t:
                if not all(var in fs[l] for l in fs):
                    continue
                m = np.isfinite(t[var])
                for l in fs:
                    m = m & np.isfinite(fs[l][var])
                if not m.any():
                    continue
                for k in range(len(names)):
                    sel = m & (rings == k)
                    if not sel.any():
                        continue
                    for l in fs:
                        d = fs[l][var][sel] - t[var][sel]
                        s = (acc[l].setdefault(var, {}).setdefault(h, {})
                             .setdefault(k, [0.0, 0.0, 0]))
                        s[0] += float(np.sum(d * d))
                        s[1] += float(np.sum(d))
                        s[2] += int(d.size)
        if args.print_every and (n + 1) % args.print_every == 0:
            print(f"    {n + 1}/{len(common)}", flush=True)

    def rmse(label, var, h, k):
        s = acc[label].get(var, {}).get(h, {}).get(k)
        return (s[0] / s[2]) ** 0.5 if s and s[2] else None

    def bias(label, var, h, k):
        """Mean signed error in the ring.

        RMSE cannot answer where an error comes from. It mixes a systematic
        offset with scatter, and the inner rings carry a far larger signal - msl
        varies by 1110 Pa inside 100 km against 440 at the rim - so a constant
        relative error reads as a much larger RMSE there whatever its origin.
        Only the signed mean separates 'the whole field is displaced' from 'the
        core is displaced'.
        """
        s = acc[label].get(var, {}).get(h, {}).get(k)
        return s[1] / s[2] if s and s[2] else None

    # Keep the order the caller asked for, dropping anything no run produced.
    wanted = sfc_names + [v + str(lev) for v, lev in upper_pairs]
    have = {v for l, *_ in runs for v in acc[l]}
    order = [v for v in wanted if v in have]
    if not order:
        raise SystemExit("no variable was present in every run at any lead")

    base = runs[0][0]
    for var in order:
        print(f"\n===== {var}, RMSE by distance from the centre =====")
        print(f"{'lead':>5}" + "".join(f"{nm + ' km':>28}" for nm in names))
        print(f"{'':>5}" + "".join(
            "".join(f"{l[:12]:>14}" for l, *_ in runs[:2]) for _ in names))
        print("-" * (5 + 28 * len(names)))
        for h in leads:
            if not any(rmse(l, var, h, 0) for l, *_ in runs):
                continue
            line = f"{h:>5}"
            for k in range(len(names)):
                for l, *_ in runs[:2]:
                    v = rmse(l, var, h, k)
                    line += f"{v:>14.4g}" if v is not None else f"{'-':>14}"
            print(line)
        # Both of the blocks below used to print a single row, at --lead, and
        # answering "how does this change with lead" meant re-running the whole
        # season once per lead - ten passes over the same files producing data
        # that one pass already had in memory. The accumulator is indexed by
        # lead throughout, so the loop is free.
        #
        # It is also the question that matters. The polar penalty is at the RIM
        # at +6 h (msl +28.7 % beyond 600 km, -6.1 % inside 100) and in the CORE
        # at +120 h (+54.8 % against +10.2 %). Those are different failures, and
        # only the shape against lead distinguishes "the boundary error is
        # propagating inward" from "one error decays while another grows".
        if args.bias:
            print("  bias (signed mean), per ring, by lead:")
            print(f"{'lead':>5} {'run':<14}"
                  + "".join(f"{nm + ' km':>14}" for nm in names))
            for h in leads:
                if all(bias(l, var, h, 0) is None for l, *_ in runs):
                    continue
                for l, *_ in runs:
                    line = f"{h:>5} {l[:14]:<14}"
                    for k in range(len(names)):
                        v = bias(l, var, h, k)
                        line += f"{v:>14.4g}" if v is not None else f"{'-':>14}"
                    print(line)
            print("    A bias flat across the rings is a whole-field offset;")
            print("    one that falls off with radius started near the centre.")
            print("    RMSE cannot tell them apart.")

        if len(runs) > 1:
            other = runs[1][0]
            print(f"  {other} against {base}, per ring, % "
                  f"(negative is better):")
            print(f"{'lead':>5}" + "".join(f"{nm + ' km':>14}" for nm in names))
            for h in leads:
                if not rmse(base, var, h, 0):
                    continue
                line = f"{h:>5}"
                for k in range(len(names)):
                    a, b = rmse(base, var, h, k), rmse(other, var, h, k)
                    line += (f"{100 * (b - a) / a:>13.1f}%"
                             if a and b else f"{'-':>14}")
                print(line)

    draw(acc, runs, names, leads, order, args)


def draw(acc, runs, names, leads, order, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = runs[0][0]
    others = [l for l, *_ in runs[1:]]
    if not others:
        print("\nonly one run; nothing to draw")
        return
    cols = min(4, len(order))
    rows = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.4 * rows),
                             squeeze=False)
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
    x = np.arange(len(names))
    for i, var in enumerate(order):
        ax = axes[i // cols][i % cols]
        for j, other in enumerate(others):
            ys = []
            for k in range(len(names)):
                a = acc[base].get(var, {}).get(args.lead, {}).get(k)
                b = acc[other].get(var, {}).get(args.lead, {}).get(k)
                if a and b and a[2] and b[2]:
                    ra = (a[0] / a[2]) ** 0.5
                    rb = (b[0] / b[2]) ** 0.5
                    # At lead 0 both runs write the initial condition verbatim,
                    # so every error is exactly zero and the ratio is 0/0. A
                    # percentage difference between two perfect scores is not a
                    # number; the panel stays blank rather than crashing.
                    ys.append(100 * (rb - ra) / ra if ra else np.nan)
                else:
                    ys.append(np.nan)
            ax.plot(x, ys, 'o-', color=colours[j % len(colours)], label=other)
        ax.axhline(0, color='0.3', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=20)
        ax.set_title(var, fontsize=10)
        ax.grid(alpha=0.3)
        if i % cols == 0:
            ax.set_ylabel(f"RMSE vs {base} [%]")
        if i == 0:
            ax.legend(fontsize=8)
    for i in range(len(order), rows * cols):
        axes[i // cols][i % cols].axis('off')
    fig.suptitle(
        f"RMSE against {base} by distance from the centre, at +{args.lead} h\n"
        f"below zero is better; the inner ring is the one the polar grid was "
        f"built for", fontsize=11)
    fig.tight_layout()
    out = os.path.expanduser(args.out or os.path.join(
        "analysis", "figures", "season",
        "radial_rmse_" + "_".join(l.replace('/', '-') for l, *_ in runs)
        + f"_{args.lead:03d}h.png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    print("\nwrote " + out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True,
                   metavar="NAME=PATH[@SELECTOR]",
                   help="the first is the reference every other is compared to")
    p.add_argument("--era5-root", required=True)
    p.add_argument("--ibtracs", default=None)
    p.add_argument("--min-vmax", type=float, default=0.0,
                   help="score only leads where the best-track Vmax at that "
                        "time is at least this, in kt. 65 is the paper's TY "
                        "group. This is the intensity AT THE FORECAST TIME, "
                        "not the storm's lifetime peak: a typhoon that has "
                        "decayed to a depression contributes its strong hours "
                        "and not its weak ones, which is what separates "
                        "'the model is bad at typhoons' from 'the sample is "
                        "mostly weak systems'. Implies --clip-to-best-track")
    p.add_argument("--clip-to-best-track", action="store_true")
    p.add_argument("--version", default=None)
    p.add_argument("--mode", default="one-way",
                   choices=["one-way", "two-way", "standalone"])
    p.add_argument("--vars", default="msl,vt10,vr10,u10,v10,t2m,tcwv,tp")
    p.add_argument("--upper", default="z:500,t:850,q:700,w:500")
    p.add_argument("--edges", default="0,100,300,600,1110",
                   help="ring boundaries in km. The defaults are the eyewall "
                        "and inner core (RMW is 33-55 km here), the inner "
                        "rainbands, the outer circulation, and the environment "
                        "out to 10 degrees where the polar disc ends")
    p.add_argument("--bias", action="store_true",
                   help="also print the signed mean per ring. RMSE mixes an "
                        "offset with scatter and the inner rings carry a much "
                        "larger signal, so it cannot say where an error "
                        "originated; the signed mean can")
    p.add_argument("--lead", type=int, default=120,
                   help="the lead the figure and the summary line use")
    p.add_argument("--max-lead", type=float, default=192)
    p.add_argument("--every", type=int, default=24)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
