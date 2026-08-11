"""Where is the vortex, as opposed to where the model says the domain is?

The polar counterpart of the coupling repository's finding_TC_center/, which
locates the storm from the weather fields - strongest 10 m wind, lowest MSLP,
peak 850 hPa vorticity - rather than from the coordinate channels. Those scripts
are hardcoded to an 81 x 81 grid with the centre at index 40 and search a fixed
[20:-20] window; everything here is derived from the data instead.

The reason to have it is not bookkeeping. The grid follows the storm, so in every
training sample - input and target alike - the vortex sits at the array centre,
and at inference the next centre is the mean of the predicted `lon`/`lat` fields
and nothing else. Two independent things therefore have to agree, and nothing
makes them:

    the position the model declares, via the coordinate channels
    the position its own weather field puts the storm at

If the vortex drifts away from the array centre, the model's dynamics and its
declared frame have come apart, and the drift is the part of the storm's motion
the coordinate channels failed to express. Adding it back gives a corrected track
at no cost - no retraining, from output already on disk.

With --era5 the same measurement is made on truth, which is the cleaner
comparison in any case: both positions then come from the same definition applied
to the same fields, instead of from a best-track agency's fix. The training data
was centred on JTWC positions and the forecasts here are verified against
JMA-derived files, and those two do not agree to better than a few tens of km.

Usage
-----
    python tools/find_center.py \
        --run "one-way=~/LLAT_polar_runs/.../start_from_2024102500" \
        --era5 /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty \
        --tc-id 202421W --init 2024102500 --out centre.png
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

DEG_KM = 111.32

# Three definitions, deliberately. They disagree by tens of km on a real storm -
# the pressure minimum is smooth but flat-bottomed, vorticity is sharp but noisy,
# the calm eye is unambiguous in a mature TC and vague in a weak one - so
# agreement between them is itself evidence the vortex is well defined, and
# disagreement is a warning not to trust any single one.
#
# Note which one is absent. The coupling repository's finding_LLAT_TC_center.py
# takes argmax of the 10 m wind speed, but the strongest wind is in the eyewall,
# not at the centre: on a perfectly axisymmetric test vortex it lands 56 km away,
# and on a real asymmetric storm it lands in the strongest quadrant, which is
# normally right of the direction of travel. That is a bias correlated with the
# very motion being measured, so the wind is used here for the calm centre it
# surrounds instead.
METHODS = ('mslp', 'vort850', 'wind_min')


def _extremum(arr, sign):
    """Index of the extremum, refined to sub-grid by a parabola on each axis.

    Raw argmax quantises the position to the grid - 0.25 degrees, 28 km - which
    is the same size as the drift being measured. The three-point parabola
    through the extremum and its neighbours costs nothing and removes it.
    """
    a = np.where(np.isfinite(arr), arr, -np.inf * sign if sign > 0 else np.inf)
    i, j = np.unravel_index(np.argmax(a * sign), a.shape)

    def shift(axis_vals):
        lo, mid, hi = axis_vals
        denom = lo - 2.0 * mid + hi
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (lo - hi) / denom, -1.0, 1.0))

    di = shift([a[i - 1, j], a[i, j], a[i + 1, j]]) if 0 < i < a.shape[0] - 1 else 0.0
    dj = shift([a[i, j - 1], a[i, j], a[i, j + 1]]) if 0 < j < a.shape[1] - 1 else 0.0
    return i + di, j + dj


def locate(field, method, search_deg):
    """(lon, lat) of the storm by one definition, searched near the array centre.

    Restricted to search_deg of the centre because the outer domain contains
    other systems - a trough, a monsoon gyre - whose wind or vorticity can exceed
    the storm's own, and because beyond a few degrees a "centre" that far from
    the frame is a tracking failure rather than a measurement.
    """
    n = field.lon.shape[0]
    c = (n - 1) / 2.0
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    near = np.hypot(xx, yy) * res <= search_deg

    if method == 'mslp':
        z, sign = field.s('msl'), -1
    elif method == 'wind_min':
        z, sign = np.hypot(field.s('u10'), field.s('v10')), -1
    elif method == 'vort850':
        z, sign = field.vorticity(850), +1
        if float(np.nanmean(field.lat)) < 0:
            sign = -1                      # cyclonic is negative south of the equator
    else:
        raise ValueError(f"unknown method {method!r}")

    z = np.where(near, np.asarray(z, dtype=float), np.nan)
    fi, fj = _extremum(z, sign)
    # Bilinear read of the coordinate fields at the sub-grid index.
    i0, j0 = int(np.floor(fi)), int(np.floor(fj))
    i0 = min(max(i0, 0), n - 2)
    j0 = min(max(j0, 0), n - 2)
    ti, tj = fi - i0, fj - j0

    def at(a):
        return float((1 - ti) * ((1 - tj) * a[i0, j0] + tj * a[i0, j0 + 1])
                     + ti * ((1 - tj) * a[i0 + 1, j0] + tj * a[i0 + 1, j0 + 1]))
    return at(field.lon), at(field.lat)


def declared(field):
    """The centre the model declares: the mean of the coordinate channels."""
    return float(np.nanmean(field.lon)), float(np.nanmean(field.lat))


def km(dlon, dlat, lat):
    return dlon * DEG_KM * np.cos(np.deg2rad(lat)), dlat * DEG_KM


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    init = datetime.datetime.strptime(args.init, "%Y%m%d%H")
    runs = [(n, os.path.expanduser(q))
            for n, q in (r.split('=', 1) for r in args.run)]
    if args.era5:
        args.era5 = os.path.expanduser(args.era5)
    for name, path in runs:
        if not os.path.isdir(os.path.expanduser(path)):
            raise SystemExit(f"run {name!r}: no such directory: {path!r}\n"
                             "(an empty path usually means a shell variable was "
                             "not set in this session)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for name, path in runs:
        meta = pf.read_meta(path)
        hours = pf.available_leads(path)

        truth = {}
        if args.era5:
            n = pf.load_run(path, hours[0], meta).sfc.shape[0]
            for h in hours:
                try:
                    truth[h] = pf.load_era5(args.era5, args.tc_id,
                                            init + datetime.timedelta(hours=h),
                                            n, meta)
                except FileNotFoundError:
                    pass

        drift, rows = {m: [] for m in METHODS}, []
        for h in hours:
            f = pf.load_run(path, h, meta)
            dec = declared(f)
            row = {'h': h, 'declared': dec}
            for m in METHODS:
                p = locate(f, m, args.search_deg)
                dx, dy = km(p[0] - dec[0], p[1] - dec[1], dec[1])
                drift[m].append(np.hypot(dx, dy))
                row[m] = (p, (dx, dy))
            rows.append(row)

        print(f"\n===== {name} =====")
        print(f"{'lead':>5} " + " ".join(f"{m:>17}" for m in METHODS))
        print(f"{'':>5} " + " ".join(f"{'offset from frame':>17}" for _ in METHODS))
        print("-" * (6 + 18 * len(METHODS)))
        for r in rows:
            if r['h'] % args.print_every:
                continue
            cells = []
            for m in METHODS:
                dx, dy = r[m][1]
                cells.append(f"{np.hypot(dx, dy):>7.0f} km {np.degrees(np.arctan2(dy, dx)):>+5.0f}°")
            print(f"{r['h']:>4}h " + " ".join(f"{c:>17}" for c in cells))

        for m in METHODS:
            d = np.asarray(drift[m])
            print(f"  {m:<8} offset from the declared frame: "
                  f"median {np.nanmedian(d):5.0f} km   max {np.nanmax(d):5.0f} km")
        print("  (a vortex that stays put means the coordinate channels carry the "
              "whole track;\n   a growing offset is motion the model produced but "
              "did not put into them)")

        line, = axes[0].plot([r['h'] for r in rows],
                             [np.hypot(*r[args.primary][1]) for r in rows],
                             'o-', lw=1.5, ms=3, label=f"{name} ({args.primary})")
        for m in METHODS:
            if m != args.primary:
                axes[0].plot([r['h'] for r in rows], drift[m], lw=1.0, alpha=0.45,
                             color=line.get_color())

        # Does using the vortex instead of the frame reduce the position error?
        if truth:
            hs = [r['h'] for r in rows if r['h'] in truth]
            e_dec, e_vor = [], []
            for r in rows:
                if r['h'] not in truth:
                    continue
                t = truth[r['h']]
                td, tv = declared(t), locate(t, args.primary, args.search_deg)
                e_dec.append(np.hypot(*km(r['declared'][0] - td[0],
                                          r['declared'][1] - td[1], td[1])))
                e_vor.append(np.hypot(*km(r[args.primary][0][0] - tv[0],
                                          r[args.primary][0][1] - tv[1], tv[1])))
            axes[1].plot(hs, e_dec, 'o-', lw=1.5, ms=3, color=line.get_color(),
                         label=f"{name}: frame vs ERA5 frame")
            axes[1].plot(hs, e_vor, 's--', lw=1.5, ms=3, color=line.get_color(),
                         label=f"{name}: vortex vs ERA5 vortex")
            print(f"  position error at the last common time: "
                  f"frame {e_dec[-1]:.0f} km, vortex {e_vor[-1]:.0f} km")

    axes[0].set_title("vortex offset from the declared domain centre", fontsize=10)
    axes[0].set_ylabel("km")
    axes[1].set_title(f"position error, two definitions ({args.primary})", fontsize=10)
    axes[1].set_ylabel("km")
    for a in axes:
        a.set_xlabel("forecast hour")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        a.set_ylim(bottom=0)
    fig.suptitle(f"{args.tc_id}  init {init:%Y-%m-%d %H}Z", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--era5", default=None)
    p.add_argument("--tc-id", default=None)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--search-deg", type=float, default=5.0,
                   help="how far from the array centre to look; the coupling "
                        "repo's scripts used a fixed [20:-20] window, which is "
                        "this at 81 x 81 and 0.25 degrees")
    p.add_argument("--primary", default='mslp', choices=METHODS,
                   help="definition used for the error comparison")
    p.add_argument("--print-every", type=int, default=24)
    p.add_argument("--out", default="centre.png")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
