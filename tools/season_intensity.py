"""Intensity skill over a whole season, with the track error held aside.

tools/intensity.py draws one case. One case cannot say whether a change helped:
202421W had P1 reaching 931.7 hPa against ERA5's 934.2 and it looked like the
patch size had solved intensity, while the season median moved by nothing.

The complication that makes a naive season mean useless: intensity error and
track error are not independent. A forecast that puts the storm five degrees
east of Taiwan keeps it over warm water, so it stays deep while the real one
lands and fills - and the resulting 40 hPa "intensity error" is a track error
wearing a different hat. That is exactly what happened to 202421W, and reporting
it as an intensity result would be wrong.

So every statistic here is reported twice: over all cases, and over the subset
whose position error at that lead is under --max-track-error. The second is the
one that answers "can this model forecast intensity", because it only counts
forecasts that are looking at roughly the right storm. The two together answer
"how much of the intensity error is really a track error", which is the question
that actually keeps coming up.

Truth is ERA5, taken from the same TC-centred files the track statistics use.
Best track is deliberately not used: JMA is 10-minute sustained wind and the
model has neither that averaging nor the resolution to reach those values, so a
bias against it measures the definition as much as the forecast.

Usage
-----
    E=/wk2/yungyun/FCNV2_TC
    python tools/season_intensity.py --mode one-way --era5-root $E \\
        --runs "cartesian=$E" \\
        --runs "baseline=$HOME/LLAT_polar_runs" \\
        --runs "t360=$HOME/LLAT_polar_runs_t360" \\
        --runs "t360_long=$HOME/LLAT_polar_runs_t360long"
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name,
                                                  os.path.join(_here, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("season_stats")
pf = ss.pf

DEG_KM = 111.32


def peak(field, search_deg):
    """Minimum MSLP in hPa and maximum 10 m wind within search_deg of centre.

    The same definition tools/intensity.py uses, so a number here and a number
    there mean the same thing. Searching a disc rather than reading the array
    centre matters because the pressure minimum drifts off the frame centre by a
    few tens of kilometres.
    """
    n = field.lon.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    m = np.hypot(xx, yy) * res <= search_deg

    msl = np.asarray(field.s('msl'), dtype=float)
    try:
        spd = np.hypot(np.asarray(field.s('u10'), float),
                       np.asarray(field.s('v10'), float))
    except ValueError:                       # a vt/vr card names them otherwise
        spd = np.hypot(np.asarray(field.s('vt10'), float),
                       np.asarray(field.s('vr10'), float))
    ok_p = m & np.isfinite(msl)
    ok_w = m & np.isfinite(spd)
    p = float(np.nanmin(msl[ok_p])) / 100.0 if ok_p.any() else np.nan
    w = float(np.nanmax(spd[ok_w])) if ok_w.any() else np.nan
    return p, w


def km(dlon, dlat, lat):
    return float(np.hypot(dlon * DEG_KM * np.cos(np.deg2rad(lat)),
                          dlat * DEG_KM))


def collect(run_dir, era5_dir, tc_id, init_str, args):
    """Per-lead (dp, dw, track error) for one case, or {} if truth is missing."""
    init = datetime.datetime.strptime(init_str, "%Y%m%d%H")
    meta = pf.read_meta(run_dir)
    out = {}
    for h in pf.available_leads(run_dir):
        if args.max_lead is not None and h > args.max_lead:
            continue
        valid = init + datetime.timedelta(hours=h)
        try:
            f = pf.load_run(run_dir, h, meta)
            n = f.lon.shape[0]
            t = pf.load_era5(era5_dir, tc_id, valid, n, meta)
        except (FileNotFoundError, OSError, KeyError):
            continue
        fp, fw = peak(f, args.search_deg)
        tp, tw = peak(t, args.search_deg)
        flon, flat = float(np.nanmean(f.lon)), float(np.nanmean(f.lat))
        tlon, tlat = float(np.nanmean(t.lon)), float(np.nanmean(t.lat))
        out[h] = (fp - tp, fw - tw, km(flon - tlon, flat - tlat, tlat))
    return out


def main(args):
    curves = {}
    era5_root = os.path.expanduser(args.era5_root)
    runs = []
    for spec in args.runs:
        label, _, path = spec.partition('=')
        runs.append((label or os.path.basename(spec.rstrip('/')), path or spec))

    # Same pairing discipline as season_stats: only cases every run produced, so
    # a run that happens to have finished more of them cannot look better by
    # covering an easier subset.
    found = {}
    for label, path in runs:
        starts = ss.find_starts(path, args.version, args.mode)
        found[label] = {(tc, init): p for tc, init, p in starts}
        print(f"  {label}: {len(found[label])} cases")
    common = set.intersection(*(set(v) for v in found.values())) \
        if not args.unpaired else None
    if common is not None:
        print(f"comparing on the {len(common)} initial times every run has; "
              f"pass --unpaired to use each run's full set")

    for label, _path in runs:
        cases = sorted(common if common is not None else found[label])
        rows = {}
        for i, (tc, init) in enumerate(cases):
            if args.limit and i >= args.limit:
                break
            r = collect(found[label][(tc, init)], os.path.join(era5_root, tc,
                                                               'ERA5',
                                                               'for_DLAMPty'),
                        tc, init, args)
            for h, v in r.items():
                rows.setdefault(h, []).append(v)
            if args.print_every and (i + 1) % args.print_every == 0:
                print(f"    {label}: {i + 1}/{len(cases)}", flush=True)

        curves[label] = rows
        print(f"\n===== {label} — {len(cases)} cases =====")
        print(f"{'lead':>5}{'n':>5}{'MSLP bias':>11}{'MSLP MAE':>10}"
              f"{'wind bias':>11}{'wind MAE':>10}"
              f"{'n<' + str(int(args.max_track_error)) + 'km':>10}"
              f"{'MSLP bias':>11}{'MSLP MAE':>10}")
        print(f"{'':>5}{'':>5}{'[hPa]':>11}{'[hPa]':>10}{'[m/s]':>11}"
              f"{'[m/s]':>10}{'':>10}{'good track':>11}{'only':>10}")
        print("-" * 83)
        for h in sorted(rows):
            if h % args.every:
                continue
            a = np.array(rows[h], dtype=float)
            dp, dw, terr = a[:, 0], a[:, 1], a[:, 2]
            g = terr <= args.max_track_error
            gp = dp[g]
            print(f"{h:>5.0f}{len(a):>5}"
                  f"{np.nanmean(dp):>11.1f}{np.nanmean(np.abs(dp)):>10.1f}"
                  f"{np.nanmean(dw):>11.1f}{np.nanmean(np.abs(dw)):>10.1f}"
                  f"{g.sum():>10}"
                  + (f"{np.nanmean(gp):>11.1f}{np.nanmean(np.abs(gp)):>10.1f}"
                     if g.sum() else f"{'-':>11}{'-':>10}"))

        if rows:
            hs = sorted(h for h in rows if h % args.every == 0)
            last = np.array(rows[hs[-1]], dtype=float)
            allb = np.nanmean(last[:, 0])
            g = last[:, 2] <= args.max_track_error
            print()
            if g.sum() >= 5:
                goodb = np.nanmean(last[g, 0])
                print(f"  At +{hs[-1]:.0f} h the mean pressure bias is "
                      f"{allb:+.1f} hPa over all {len(last)} cases and "
                      f"{goodb:+.1f} hPa")
                print(f"  over the {g.sum()} whose track error is under "
                      f"{args.max_track_error:.0f} km.")
                if abs(allb) > abs(goodb) + 2:
                    print("  Most of the apparent intensity error is a track "
                          "error: the")
                    print("  forecasts that went to the right place are much "
                          "closer.")
                else:
                    print("  The bias survives restricting to good tracks, so "
                          "it is an")
                    print("  intensity error in its own right.")
            else:
                print(f"  Only {g.sum()} cases have a track error under "
                      f"{args.max_track_error:.0f} km at +{hs[-1]:.0f} h, too "
                      f"few to separate\n  intensity skill from track error at "
                      f"this lead.")

    draw(curves, runs, args)


def draw(curves, runs, args):
    """Four panels, the fourth of which is not about intensity at all.

    MAE, bias and wind are the intensity result. The count of cases inside
    --max-track-error is a track-skill measure that comes free with the same
    pass, and it is the panel that carries the season's clearest signal:
    t360_long keeps 72 of 81 cases inside 200 km at +24 h against the baseline's
    53. It also says how far each of the other panels can be trusted - a
    good-track statistic computed over two cases is not a statistic.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = os.path.expanduser(args.out or os.path.join(
        "analysis", "figures", "season",
        "intensity_" + "_".join(l.replace('/', '-') for l, _ in runs) + ".png"))

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    for label, _ in runs:
        rows = curves.get(label) or {}
        hs = sorted(h for h in rows if h % args.every == 0)
        if not hs:
            continue
        a = {h: np.array(rows[h], dtype=float) for h in hs}
        mae = [np.nanmean(np.abs(a[h][:, 0])) for h in hs]
        bias = [np.nanmean(a[h][:, 0]) for h in hs]
        wmae = [np.nanmean(np.abs(a[h][:, 1])) for h in hs]
        ngood = [int((a[h][:, 2] <= args.max_track_error).sum()) for h in hs]
        # The good-track series is drawn dashed and only where enough cases
        # survive to mean anything; below that it is one or two storms.
        gmae = [np.nanmean(np.abs(a[h][a[h][:, 2] <= args.max_track_error, 0]))
                if (a[h][:, 2] <= args.max_track_error).sum() >= args.min_good
                else np.nan for h in hs]
        ln, = ax[0].plot(hs, mae, '-o', ms=3, label=label)
        ax[0].plot(hs, gmae, '--', color=ln.get_color(), lw=1.2)
        ax[1].plot(hs, bias, '-o', ms=3, color=ln.get_color(), label=label)
        ax[2].plot(hs, wmae, '-o', ms=3, color=ln.get_color(), label=label)
        ax[3].plot(hs, ngood, '-o', ms=3, color=ln.get_color(), label=label)

    ax[0].set_title(f"MSLP mean absolute error\nsolid all cases, dashed track "
                    f"< {args.max_track_error:.0f} km")
    ax[0].set_ylabel("hPa")
    ax[1].set_title("MSLP bias (forecast - ERA5)\nnegative means too deep")
    ax[1].set_ylabel("hPa")
    ax[1].axhline(0, color='0.5', lw=0.8)
    ax[2].set_title("10 m wind mean absolute error")
    ax[2].set_ylabel("m s$^{-1}$")
    ax[3].set_title(f"cases within {args.max_track_error:.0f} km\n"
                    f"track skill, and how far the dashed lines can be trusted")
    ax[3].set_ylabel("cases")
    for a_ in ax:
        a_.set_xlabel("forecast hour")
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True,
                   metavar="[NAME=]PATH")
    p.add_argument("--era5-root", required=True)
    p.add_argument("--version", default=None)
    p.add_argument("--mode", default="one-way",
                   choices=("one-way", "two-way", "standalone"))
    p.add_argument("--search-deg", type=float, default=5.0,
                   help="radius searched for the pressure minimum and wind "
                        "maximum; matches tools/intensity.py")
    p.add_argument("--max-track-error", type=float, default=200.0,
                   help="a forecast further than this from the analysed centre "
                        "is looking at a different storm, and its intensity "
                        "error is mostly a track error")
    p.add_argument("--every", type=int, default=24,
                   help="print every Nth forecast hour")
    p.add_argument("--max-lead", type=float, default=120)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--unpaired", action="store_true")
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--min-good", type=int, default=8,
                   help="do not draw the good-track curve where fewer cases "
                        "than this survive the track filter; two storms is not "
                        "a statistic and a line drawn through them reads as one")
    p.add_argument("--out", default=None,
                   help="figure path; defaults to analysis/figures/season/"
                        "intensity_<labels>.png, named after the runs so a "
                        "second comparison does not overwrite the first")
    main(p.parse_args())
