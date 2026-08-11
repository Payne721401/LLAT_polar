"""Storm intensity through the forecast: how deep, how strong, and against what.

Track says where the storm is; this says what it is. The reference paper reports
systematic under-prediction of intensity for severe typhoons, and concentrating
resolution in the inner core was one of the two reasons for going to a polar grid
at all, so this is the number that says whether that worked.

Three curves, and the difference between them is the point.

    the forecast          minimum MSLP and maximum 10 m wind near the centre
    ERA5                  the same measurement on the analysis the model was
                          trained against - the fair comparison, and the ceiling
                          on what the model could possibly reproduce
    best track            what the storm actually was

ERA5 at 0.25 deg cannot resolve an eyewall. A 900 hPa typhoon appears in it
around 950-960, and its 10 m wind maximum is far below the observed one. So a
forecast that matches ERA5 is doing everything that can be asked of it, while the
gap between ERA5 and best track is a limit of the training data, not of the model.
Plotting only against best track blames the model for both.

The wind comparison is looser still: best-track wind is a sustained value over a
1- or 10-minute window depending on the agency, while ERA5 and the model give an
instantaneous value averaged over a 0.25 deg cell. Read the shape of the curve -
does the forecast deepen when the storm deepened - rather than the offset.

Usage
-----
    python tools/intensity.py \
        --run "1.0=~/LLAT_polar_runs/.../start_from_2024102700" \
        --run "1.45=~/LLAT_polar_runs/..._scale1.45/start_from_2024102700" \
        --era5 /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty \
        --track-csv /wk2/yungyun/ERA5_2024_for_TC/TC_list_JMA_v2 \
        --tc-id 202421W --init 2024102700 --out intensity.png
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "plot_forecast", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "plot_forecast.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

KT_TO_MS = 0.514444


def near_centre(field, search_deg):
    """Cells within search_deg of the array centre.

    The storm is at the centre by construction, and the outer domain routinely
    holds a deeper low - a mid-latitude trough at the edge of a 20 degree square
    is entirely normal - which would otherwise be reported as the TC.
    """
    n = field.lon.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    return np.hypot(xx, yy) * res <= search_deg


def measure(field, search_deg):
    """(minimum MSLP in hPa, maximum 10 m wind in m/s) near the centre."""
    m = near_centre(field, search_deg)
    msl = np.asarray(field.s('msl'), dtype=float)[m] / 100.0
    ws = np.hypot(np.asarray(field.s('u10'), dtype=float),
                  np.asarray(field.s('v10'), dtype=float))[m]
    return float(np.nanmin(msl)), float(np.nanmax(ws))


def best_track(csv_dir, tc_id, init, hours):
    """Observed pressure and wind at each forecast hour, where a record exists."""
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.expanduser(csv_dir), f"{tc_id}.csv"))
    df['t'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    by_time = {t: row for t, row in zip(df['t'], df.to_dict('records'))}
    out = {}
    for h in hours:
        row = by_time.get(init + datetime.timedelta(hours=h))
        if row is not None:
            out[h] = (float(row['Pressure (hPa)']),
                      float(row['Wind (kt)']) * KT_TO_MS)
    return out


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    init = datetime.datetime.strptime(args.init, "%Y%m%d%H")
    runs = [(n, os.path.expanduser(q))
            for n, q in (r.split('=', 1) for r in args.run)]
    for name, path in runs:
        if not os.path.isdir(path):
            raise SystemExit(f"run {name!r}: no such directory: {path!r}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    print(f"{'source':<12} {'lead':>5} {'MSLP':>9} {'10 m wind':>11}")
    print("-" * 42)

    for name, path in runs:
        meta = pf.read_meta(path)
        hours = pf.available_leads(path)
        vals = [measure(pf.load_run(path, h, meta), args.search_deg) for h in hours]
        p, w = zip(*vals)
        line, = axes[0].plot(hours, p, 'o-', lw=1.5, ms=3, label=name)
        axes[1].plot(hours, w, 'o-', lw=1.5, ms=3, color=line.get_color(), label=name)
        for h, (pp, ww) in zip(hours, vals):
            if h % args.print_every == 0:
                print(f"{name:<12} {h:>4}h {pp:>8.1f} {ww:>10.1f}")
        print(f"{'':<12} deepest {min(p):.1f} hPa, strongest {max(w):.1f} m/s")

    if args.era5:
        _, path = runs[0]
        meta = pf.read_meta(path)
        n = pf.load_run(path, pf.available_leads(path)[0], meta).sfc.shape[0]
        hs, vals = [], []
        for h in pf.available_leads(path):
            try:
                f = pf.load_era5(os.path.expanduser(args.era5), args.tc_id,
                                 init + datetime.timedelta(hours=h), n, meta)
            except FileNotFoundError:
                continue                  # ERA5 is 6-hourly; LLAT steps 3 h
            hs.append(h)
            vals.append(measure(f, args.search_deg))
        if hs:
            p, w = zip(*vals)
            axes[0].plot(hs, p, 'k-', lw=2.2, label='ERA5')
            axes[1].plot(hs, w, 'k-', lw=2.2, label='ERA5')
            print(f"{'ERA5':<12} deepest {min(p):.1f} hPa, "
                  f"strongest {max(w):.1f} m/s")

    if args.track_csv:
        bt = best_track(args.track_csv, args.tc_id, init,
                        pf.available_leads(runs[0][1]))
        if bt:
            hs = sorted(bt)
            axes[0].plot(hs, [bt[h][0] for h in hs], 's--', c='0.45', lw=1.6,
                         ms=4, label='best track')
            axes[1].plot(hs, [bt[h][1] for h in hs], 's--', c='0.45', lw=1.6,
                         ms=4, label='best track')
            print(f"{'best track':<12} deepest {min(v[0] for v in bt.values()):.1f}"
                  f" hPa, strongest {max(v[1] for v in bt.values()):.1f} m/s")
            print("\nbest track is a sustained wind over a 1- or 10-minute window; "
                  "ERA5 and the\nmodel give an instantaneous 0.25 deg cell mean. "
                  "The gap between the grey and\nblack curves is what the training "
                  "data cannot represent, not a model error.")

    axes[0].invert_yaxis()                # deeper is stronger, so it reads upward
    axes[0].set_ylabel("minimum MSLP [hPa]")
    axes[1].set_ylabel("maximum 10 m wind [m s$^{-1}$]")
    for a in axes:
        a.set_xlabel("forecast hour")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{args.tc_id}  init {init:%Y-%m-%d %H}Z  "
                 f"within {args.search_deg:g}° of the centre", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--era5", default=None,
                   help="adds the analysis the model was trained against, which "
                        "is the fair comparison and the ceiling on what it could "
                        "reproduce")
    p.add_argument("--track-csv", default=None,
                   help="adds observed pressure and wind, which shows how much "
                        "of the gap belongs to ERA5 rather than to the model")
    p.add_argument("--tc-id", required=True)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--search-deg", type=float, default=5.0)
    p.add_argument("--print-every", type=int, default=24)
    p.add_argument("--out", default="intensity.png")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
