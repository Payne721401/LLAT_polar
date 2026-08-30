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

def clip_times(args, tc_id, init_str, run_dir):
    """Valid times with an IBTrACS record, or None when not clipping.

    The ERA5 boxes these tools read were cut along a track that outlives the
    agency's: 202408W's run 3.5 days past its last best-track entry, following a
    1010.8 hPa remnant to 40N and the dateline. Scoring those leads measures how
    well a model chases a system that no longer exists, and on the track
    statistics it hid a 20-28 % difference between the two models. The paper's
    rule is one sample per best-track record, and its archive is IBTrACS.
    """
    if not getattr(args, "clip_to_best_track", False):
        return None
    sid = ss.storm_sid(args.ibtracs, tc_id, init_str, run_dir)
    return ss.ibt.times(args.ibtracs, sid) if sid else set()

pf = ss.pf

DEG_KM = 111.32

KT = 0.514444          # knots to m/s

# The paper's grouping (Fig. 8), on best-track Vmax at the forecast time, not at
# the storm's peak: TD below 35 kt, TS in [35, 65), TY at or above 65. It
# reports TD track errors about 30 % larger and a 1.5x wider spread, and
# separately that LLAT.ty underpredicts intensity for TYs while being bias-free
# for TDs and TSs. Neither statement can be checked against an all-cases mean.
BINS = [(0.0, 35.0, "TD <35"), (35.0, 65.0, "TS 35-65"), (65.0, 1e9, "TY >=65")]


def bin_of(vmax):
    if vmax is None or not np.isfinite(vmax):
        return None
    for lo, hi, name in BINS:
        if lo <= vmax < hi:
            return name
    return None



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
    """Per-lead (dp, dw, track error, best-track Vmax) for one case.

    The last field is what the sample is binned on and is never the model's own
    intensity: the paper conditions on best-track Vmax at the forecast time, so
    a bin has to mean the same thing for every run being compared.

    With --truth best the pressure is compared against IBTrACS rather than
    against ERA5's own minimum. That is not a cosmetic change. ERA5 cannot
    resolve an eyewall at 0.25 degrees and is systematically too weak for
    intense storms - it is the reason the paper reports a +40 hPa weak bias for
    best-track MSLP below 950 - so "too deep against ERA5" and "too deep against
    best track" are different claims, and only the second can be set beside the
    paper's.

    The wind comparison stays informational under --truth best: best-track Vmax
    is a 1-minute sustained estimate and the model gives an instantaneous
    0.25-degree cell maximum, which are not the same quantity.
    """
    init = datetime.datetime.strptime(init_str, "%Y%m%d%H")
    meta = pf.read_meta(run_dir)
    clip = clip_times(args, tc_id, init_str, run_dir)
    sid = (ss.storm_sid(args.ibtracs, tc_id, init_str, run_dir)
           if args.ibtracs else None)
    rec = ss.ibt.load(args.ibtracs)[sid]["rec"] if sid else {}
    out = {}
    for h in pf.available_leads(run_dir):
        if args.max_lead is not None and h > args.max_lead:
            continue
        valid = init + datetime.timedelta(hours=h)
        if clip is not None and valid not in clip:
            continue
        try:
            # peak() reads msl and collect() reads lon/lat; nothing here touches
            # the upper air, which is 81 % of both files.
            f = pf.load_run(run_dir, h, meta, upper=False)
            n = f.lon.shape[0]
            t = pf.load_era5(era5_dir, tc_id, valid, n, meta, upper=False)
        except (FileNotFoundError, OSError, KeyError):
            continue
        fp, fw = peak(f, args.search_deg)
        flon, flat = float(np.nanmean(f.lon)), float(np.nanmean(f.lat))
        tlon, tlat = float(np.nanmean(t.lon)), float(np.nanmean(t.lat))
        b = rec.get(valid)
        if args.truth == "best":
            if not b or b["pres"] is None:
                continue
            tp = b["pres"]
            tw = b["vmax"] * KT if b["vmax"] is not None else np.nan
        else:
            tp, tw = peak(t, args.search_deg)
        out[h] = (fp - tp, fw - tw, km(flon - tlon, flat - tlat, tlat),
                  b["vmax"] if b and b["vmax"] is not None else np.nan)
    return out


def report_strat(rows, label, args):
    """MSLP bias split by best-track Vmax at the forecast time."""
    hs = sorted(h for h in rows if h % args.every == 0)
    names = [n for _, _, n in BINS]
    print()
    print(f"----- {label}: MSLP bias by best-track intensity at the "
          f"forecast time -----")
    print(f"{'lead':>5}" + "".join(f"{n:>20}" for n in names))
    print(f"{'':>5}" + "".join(f"{'n':>8}{'bias [hPa]':>12}" for _ in names))
    print("-" * (5 + 20 * len(names)))
    for h in hs:
        a = np.array(rows[h], dtype=float)
        line = f"{h:>5.0f}"
        for _lo, _hi, n in BINS:
            m = np.array([bin_of(v) == n for v in a[:, 3]])
            if m.sum() < args.min_bin:
                line += f"{m.sum():>8}{'-':>12}"
            else:
                line += f"{m.sum():>8}{np.nanmean(a[m, 0]):>12.1f}"
        print(line)
    print(f"  bins with fewer than {args.min_bin} samples are left blank; the "
          f"paper's own\n  intensity claim is about the TY column alone.")


def draw_strat(curves, runs, args):
    """One panel per best-track intensity group, MSLP bias against lead.

    This is the figure the result lives in, and it is deliberately plainer than
    the eight-panel overview: four runs with a median, a mean and a shaded
    interquartile range is twelve overlapping things per panel, and the reader
    cannot see which line is above which. Here it is one line per run, a zero
    line, and the sample count in the label - because the whole claim is about
    which side of zero each group sits on and by how much.

    Positive is too weak, which is the sign convention the paper reports its
    +40 hPa in. Splitting matters because the groups disagree: averaged
    together, a strongly positive TY column and a strongly negative TD column
    make a mean that describes neither.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [n for _, _, n in BINS]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.4),
                             sharex=True)
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax, (lo, hi, name) in zip(axes, BINS):
        for (label, *_), c in zip(runs, colours * 4):
            rows = curves[label]
            hs = sorted(h for h in rows if h % args.every == 0)
            xs, ys, ns = [], [], []
            for h in hs:
                a = np.array(rows[h], dtype=float)
                m = np.array([bin_of(v) == name for v in a[:, 3]])
                if m.sum() < args.min_bin:
                    continue
                xs.append(h)
                ys.append(float(np.nanmean(a[m, 0])))
                ns.append(int(m.sum()))
            if xs:
                ax.plot(xs, ys, 'o-', ms=4, lw=1.8, color=c,
                        label=f"{label} (n {ns[0]}→{ns[-1]})")
        ax.axhline(0, color='0.3', lw=1)
        ax.set_title(f"{name} kt", fontsize=11)
        ax.set_xlabel("forecast hour")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MSLP bias [hPa]   positive = too weak")

    fig.suptitle("MSLP bias against best track, split by best-track Vmax at the "
                 "forecast time\n"
                 "the groups disagree in sign, so an all-cases mean describes "
                 "neither", fontsize=11)
    fig.tight_layout()
    out = os.path.expanduser(args.strat_out or os.path.join(
        "analysis", "figures", "season",
        "intensity_strat_" + "_".join(l.replace('/', '-') for l, *_ in runs)
        + ".png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi if hasattr(args, "dpi") else 150,
                bbox_inches='tight')
    print("\nwrote " + out)


def main(args):
    curves = {}
    era5_root = os.path.expanduser(args.era5_root)
    # ss.parse_run: a spec may name its own mode or model directory, so one
    # figure can hold one-way and two-way together on the same paired cases.
    runs = [ss.parse_run(spec, args.mode) for spec in args.runs]

    # Same pairing discipline as season_stats: only cases every run produced, so
    # a run that happens to have finished more of them cannot look better by
    # covering an easier subset.
    found = {}
    for label, path, mode, subdir in runs:
        starts = ss.find_starts(path, args.version, mode, subdir)
        found[label] = {(tc, init): p for tc, init, p in starts}
        print(f"  {label}: {len(found[label])} cases")
    common = set.intersection(*(set(v) for v in found.values())) \
        if not args.unpaired else None
    if common is not None:
        print(f"comparing on the {len(common)} initial times every run has; "
              f"pass --unpaired to use each run's full set")

    for label, *_ in runs:
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
        if args.strat and rows:
            report_strat(rows, label, args)
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
    if args.strat:
        draw_strat(curves, runs, args)


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
        "intensity_" + "_".join(l.replace('/', '-') for l, *_ in runs) + ".png"))

    def band(a_, hs, samples, colour, label, min_n=1):
        """One series in season_stats' convention: median, IQR, mean.

        Solid median, shaded interquartile range, dashed mean. The median is the
        headline because a season of forecasts is not normally distributed - two
        storms that went badly wrong drag a mean a long way - and where the
        dashed line separates from the solid one, a few cases are carrying it.
        Drawn nowhere the sample falls below min_n, since a median over two
        cases is a number rather than a statistic.
        """
        med, lo, hi, mean = [], [], [], []
        for s in samples:
            s = s[np.isfinite(s)]
            if len(s) < min_n:
                med.append(np.nan), lo.append(np.nan)
                hi.append(np.nan), mean.append(np.nan)
                continue
            med.append(np.median(s)), mean.append(np.mean(s))
            lo.append(np.percentile(s, 25)), hi.append(np.percentile(s, 75))
        ln, = a_.plot(hs, med, '-o', ms=3, color=colour, label=label)
        a_.plot(hs, mean, '--', color=ln.get_color(), lw=1.0)
        a_.fill_between(hs, lo, hi, color=ln.get_color(), alpha=0.13, lw=0)
        return ln.get_color()

    fig, ax = plt.subplots(2, 4, figsize=(19, 8.6), squeeze=False)
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for (label, *_), c in zip(runs, colours * 4):
        rows = curves.get(label) or {}
        hs = sorted(h for h in rows if h % args.every == 0)
        if not hs:
            continue
        a = {h: np.array(rows[h], dtype=float) for h in hs}
        good = {h: a[h][a[h][:, 2] <= args.max_track_error] for h in hs}

        # Top row: every case. Bottom row: only those looking at roughly the
        # right storm, which is the row that answers "can it forecast
        # intensity" rather than "did it go to the right place".
        band(ax[0][0], hs, [np.abs(a[h][:, 0]) for h in hs], c, label)
        band(ax[0][1], hs, [a[h][:, 0] for h in hs], c, label)
        band(ax[0][2], hs, [np.abs(a[h][:, 1]) for h in hs], c, label)
        ax[0][3].plot(hs, [len(a[h]) for h in hs], '-o', ms=3, color=c,
                      label=label)

        band(ax[1][0], hs, [np.abs(good[h][:, 0]) for h in hs], c, label,
             args.min_good)
        band(ax[1][1], hs, [good[h][:, 0] for h in hs], c, label, args.min_good)
        band(ax[1][2], hs, [np.abs(good[h][:, 1]) for h in hs], c, label,
             args.min_good)
        ax[1][3].plot(hs, [len(good[h]) for h in hs], '-o', ms=3, color=c,
                      label=label)

    ax[0][0].set_title("MSLP absolute error — all cases\nsolid median, shaded "
                       "IQR, dashed mean")
    ax[0][1].set_title("MSLP bias (forecast - ERA5)\nnegative means too deep")
    ax[0][2].set_title("10 m wind absolute error")
    ax[0][3].set_title("cases contributing\nstorms end, and the survivors are "
                       "not a random subset")
    ax[1][0].set_title(f"MSLP absolute error — track < "
                       f"{args.max_track_error:.0f} km\nintensity skill, with "
                       f"the track error held aside")
    ax[1][1].set_title(f"MSLP bias — track < {args.max_track_error:.0f} km")
    ax[1][2].set_title(f"10 m wind error — track < "
                       f"{args.max_track_error:.0f} km")
    ax[1][3].set_title(f"cases within {args.max_track_error:.0f} km\n"
                       f"track skill, and how far this row can be trusted")
    for r in (0, 1):
        ax[r][0].set_ylabel("hPa"), ax[r][1].set_ylabel("hPa")
        ax[r][2].set_ylabel("m s$^{-1}$"), ax[r][3].set_ylabel("cases")
        ax[r][1].axhline(0, color='0.5', lw=0.8)
        for a_ in ax[r]:
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
    p.add_argument("--ibtracs", default=None, metavar="CSV",
                   help="IBTrACS basin file, for --clip-to-best-track. Storms "
                        "are matched by position and time, never by number")
    p.add_argument("--truth", default="era5", choices=["era5", "best"],
                   help="era5 compares the model's MSLP minimum with ERA5's "
                        "own; best compares it with the IBTrACS central "
                        "pressure, which is what the paper verifies against. "
                        "ERA5 cannot resolve an eyewall at 0.25 deg and is too "
                        "weak for intense storms - that is the source of the "
                        "paper's +40 hPa - so the two answer different "
                        "questions and only the second is comparable with it")
    p.add_argument("--strat", action="store_true",
                   help="split the MSLP bias by best-track Vmax at the "
                        "forecast time, into the paper's TD / TS / TY groups. "
                        "Its intensity claim is about the TY group alone and "
                        "an all-cases mean cannot be set against it")
    p.add_argument("--strat-out", default=None,
                   help="figure path for the stratified panels; defaults to "
                        "analysis/figures/season/intensity_strat_<labels>.png")
    p.add_argument("--min-bin", type=int, default=10,
                   help="a stratified cell with fewer samples than this prints "
                        "blank rather than a number nobody should read")
    p.add_argument("--clip-to-best-track", action="store_true",
                   help="score only leads with an IBTrACS record. Without it "
                        "the ERA5 boxes carry the storm days past the last "
                        "entry and those leads are graded like any other; that "
                        "hid a 20-28 %% difference between the models on track")
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
