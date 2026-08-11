"""The global model's own track, from output a coupled run already saved.

In one-way mode FCNV2 is never written back to - that is what one-way means - so
the `FCNV2/forecast/output_weather_*.npy` files a coupled run leaves behind are a
free-running FCNV2 forecast. Its track is therefore already on disk and costs
nothing to extract, and it is the reference the paper's Fig. 4b draws in yellow.

Do not use a two-way run for this. There LLAT's inner structure is fed back into
the global state every 6 h, so what is saved is no longer FCNV2 alone.

Following the coupling repository's finding_FCNV2_TC_center.py, the centre is the
minimum mean sea level pressure - channel 6 of the 73 - searched near the previous
centre rather than globally, because the saved sub-domain reaches 80 N and a
mid-latitude cyclone there is deeper than any tropical one.

Usage
-----
    python tools/fcnv2_track.py \
        --run ~/LLAT_polar_runs/202421W/one_way_couple_model_.../start_from_2024102500 \
        --era5 /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty \
        --tc-id 202421W --init 2024102500
"""
import argparse
import datetime
import glob
import importlib.util
import os
import re

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "plot_forecast", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "plot_forecast.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

DEG_KM = 111.32
MSL_CHANNEL = 6

# The sub-domain run_coupled_forecast saves, as WP_BOX resolves on the 721 x 1440
# FCNV2 grid: 80 N down to 10 S, 80 E to 180 E, latitude descending.
LAT_MAX, LAT_MIN, LON_MIN, LON_MAX = 80.0, -10.0, 80.0, 180.0


def grid(shape):
    """Latitude and longitude of the saved sub-domain, from its shape."""
    ny, nx = shape
    return (np.linspace(LAT_MAX, LAT_MIN, ny),
            np.linspace(LON_MIN, LON_MAX, nx))


def leads(run_dir):
    out = []
    for p in glob.glob(os.path.join(run_dir, 'FCNV2', 'forecast',
                                    'output_weather_*h.npy')):
        m = re.search(r'output_weather_(\d+)h\.npy$', p)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def centre(arr, prev, search_deg):
    """Minimum MSLP within search_deg of the previous centre."""
    msl = arr[MSL_CHANNEL]
    lat, lon = grid(msl.shape)
    lon2d, lat2d = np.meshgrid(lon, lat)
    d = np.hypot((lon2d - prev[0]) * np.cos(np.deg2rad(prev[1])), lat2d - prev[1])
    z = np.where(d <= search_deg, msl, np.inf)
    i, j = np.unravel_index(np.argmin(z), z.shape)
    return float(lon[j]), float(lat[i]), float(msl[i, j]) / 100.0


def main(args):
    run = os.path.expanduser(args.run)
    hs = leads(run)
    if not hs:
        raise SystemExit(
            f"no FCNV2/forecast/output_weather_*.npy under {run!r}. "
            "standalone runs have none, and a two-way run would not be a free run.")

    init = datetime.datetime.strptime(args.init, "%Y%m%d%H")
    truth, n = {}, None
    if args.era5:
        for h in hs:
            try:
                f = pf.load_era5(os.path.expanduser(args.era5), args.tc_id,
                                 init + datetime.timedelta(hours=h),
                                 n or 81, None)
            except FileNotFoundError:
                continue
            truth[h] = (float(np.nanmean(f.lon)), float(np.nanmean(f.lat)))

    if truth:
        prev = truth[min(truth)]
    elif args.start:
        prev = tuple(float(v) for v in args.start.split(','))
    else:
        raise SystemExit("give --era5 or --start LON,LAT so the search has a seed")

    print(f"{'lead':>5} {'position':>20} {'MSLP':>9} {'error':>9}")
    print("-" * 47)
    track = {}
    for h in hs:
        arr = np.load(os.path.join(run, 'FCNV2', 'forecast',
                                   f'output_weather_{h:0>3}h.npy'))
        lon, lat, msl = centre(arr, prev, args.search_deg)
        prev = (lon, lat)
        track[h] = (lon, lat)
        err = ''
        if h in truth:
            ex = (lon - truth[h][0]) * DEG_KM * np.cos(np.deg2rad(truth[h][1]))
            ey = (lat - truth[h][1]) * DEG_KM
            err = f"{np.hypot(ex, ey):>8.0f} km"
        print(f"{h:>4}h {lon:>9.2f}E {lat:>6.2f}N {msl:>8.1f} {err:>9}")

    lo, la = zip(*[track[h] for h in hs])
    print(f"\nFCNV2 free run: {lo[0]:.2f}E,{la[0]:.2f}N -> {lo[-1]:.2f}E,{la[-1]:.2f}N")
    if truth:
        t = [truth[h] for h in sorted(truth)]
        print(f"ERA5          : {t[0][0]:.2f}E,{t[0][1]:.2f}N -> "
              f"{t[-1][0]:.2f}E,{t[-1][1]:.2f}N")

    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot(lo, la, '-', lw=1.6, c='tab:orange', label='FCNV2 free run')
        daily = [i for i, h in enumerate(hs) if h % 24 == 0]
        ax.plot([lo[i] for i in daily], [la[i] for i in daily], 'o', ms=5,
                c='tab:orange')
        if truth:
            th = sorted(truth)
            tl, ta = zip(*[truth[h] for h in th])
            ax.plot(tl, ta, 'k-', lw=2.2, label='ERA5')
            ax.plot(tl[0], ta[0], 'k*', ms=18, zorder=5)
            ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(ta))))
        ax.set_xlabel("longitude [°E]")
        ax.set_ylabel("latitude [°N]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.suptitle(f"{args.tc_id}  init {init:%Y-%m-%d %H}Z", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a ONE-WAY start_from_* directory")
    p.add_argument("--era5", default=None)
    p.add_argument("--tc-id", required=True)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--start", default=None, metavar="LON,LAT",
                   help="seed for the search when --era5 is not given")
    p.add_argument("--search-deg", type=float, default=5.0,
                   help="how far from the previous centre to look; the saved box "
                        "reaches 80 N, where an extratropical low is deeper than "
                        "any TC")
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
