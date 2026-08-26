"""Forecast track against ERA5: position error, and how it decomposes.

For a TC model this is the number that matters, and nothing here produced it yet.
The domain centre IS the forecast position - the grid follows the storm, so the
mean of the lon/lat channels is where the model thinks the storm is - and the
ERA5 files are centred on the analysed position at the same valid time. Both
tracks therefore fall out of data already on disk.

Reports total error and splits it into along-track and cross-track. That split is
worth the extra lines: a model that goes the right way too slowly and one that
goes the wrong way can post the same total error and need completely different
fixes. Alongside them it draws the tracks themselves, because even the split can
mislead - a forecast heading somewhere else entirely will cross the observed
track on its way past and post a small error at that moment.

Usage
-----
    python tools/track_error.py \
        --run "one-way=~/runs/.../start_from_2024102500" \
        --era5 /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty \
        --tc-id 202421W --init 2024102500 --out track.png
"""
import argparse
import csv
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


def read_best_track(csv_dir, tc_id, want="pressure"):
    """{datetime: value} from the best-track CSV, if it carries the column.

    want="pressure" gives central pressure in hPa, want="position" gives
    (lon, lat) in degrees. One function because the hard parts - finding the
    columns and rebuilding the timestamp - are the same either way, and two
    copies of them would drift.

    Returns {} when the file has no such column, which every caller reads as
    "this file cannot answer that".
    """
    p = os.path.join(os.path.expanduser(csv_dir), tc_id + ".csv")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    keys = {k.lower().strip(): k for k in rows[0]}

    def find(exact, substrings):
        """Exact column name first, then a substring pass."""
        for k in exact:
            if k in keys:
                return keys[k]
        for k in keys:
            if any(t in k for t in substrings):
                return keys[k]
        return None

    # The JMA lists this project uses head their columns "Pressure (hPa)",
    # "Lat." and "Long.", and split the time across Year, Month, Day and Hour.
    # None of that matched the exact names the first version looked for, so this
    # returned {} - which callers read as "no pressure in this file". --truth
    # best was therefore silently absent from every landfall table rather than
    # failing. The data was there throughout.
    if want == "position":
        cols = [find(("lon", "longitude", "long"), ("lon",)),
                find(("lat", "latitude"), ("lat",))]
    else:
        cols = [find(("mslp", "pres", "pressure", "min_pres", "cp", "slp"),
                     ("mslp", "pressure", "pres", "slp"))]
    tkey = find(("time", "date", "datetime", "yyyymmddhh"), ())
    ymdh = [keys.get(k) for k in ("year", "month", "day", "hour")]
    if not all(cols) or not (tkey or all(ymdh)):
        return {}

    def value(r):
        v = tuple(float(r[c]) for c in cols)
        return v if len(v) > 1 else v[0]

    for r in rows:
        if not tkey:
            try:
                t = datetime.datetime(*(int(float(r[c])) for c in ymdh))
                out[t] = value(r)
            except (TypeError, ValueError):
                pass
            continue
        raw = str(r[tkey]).strip().replace("-", "").replace(":", "") \
            .replace(" ", "").replace("/", "")
        # Match the format to the string's LENGTH. Trying the longest first
        # and catching ValueError is not enough: strptime accepts a short
        # string against a long format in some builds and returns a silently
        # wrong date - 2024102700 came back as 2024-10-02 07:00.
        fmt = {14: "%Y%m%d%H%M%S", 12: "%Y%m%d%H%M", 10: "%Y%m%d%H",
               8: "%Y%m%d"}.get(len(raw))
        if fmt is None:
            continue
        try:
            t = datetime.datetime.strptime(raw, fmt)
        except ValueError:
            continue
        try:
            out[t] = value(r)
        except (TypeError, ValueError):
            pass
    return out


def centre(field):
    """Where the model thinks the storm is.

    nanmean over the disc: the corners are NaN by design, and the remaining
    region is still symmetric about the centre, which is what makes the mean
    equal the centre in the first place.
    """
    return float(np.nanmean(field.lon)), float(np.nanmean(field.lat))


def km(dlon, dlat, lat):
    """Degrees to km, with the cos(lat) factor on longitude."""
    return (dlon * DEG_KM * np.cos(np.deg2rad(lat)), dlat * DEG_KM)


def draw_track(ax, hours, lons, lats, colour, label, lw=1.6):
    """One track on the map, marked every 24 h.

    The daily markers are what make lag readable: two tracks can lie on top of
    each other while the storms sit a day apart along them, and that is exactly
    the failure mode here.
    """
    ax.plot(lons, lats, '-', lw=lw, color=colour, label=label)
    daily = [i for i, h in enumerate(hours) if h % 24 == 0]
    ax.plot([lons[i] for i in daily], [lats[i] for i in daily],
            'o', ms=5, color=colour)


def _prepare(out):
    """Make the parent directory of an output path, so --out can organise.

    Figures belong under analysis/figures/, filed by case and initial time, and
    requiring the directory to exist first turns every plotting command into two.
    """
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return out


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    init = datetime.datetime.strptime(args.init, "%Y%m%d%H")
    runs = [(n, os.path.expanduser(q))
            for n, q in (r.split('=', 1) for r in args.run)]
    args.era5 = os.path.expanduser(args.era5)
    # An unset shell variable expands to nothing, so --run "one-way=$O" becomes
    # --run "one-way=" and every later step quietly finds no files. Saying so
    # here beats "no ERA5 file matched any forecast hour" ten lines further on.
    for name, path in runs:
        if not os.path.isdir(os.path.expanduser(path)):
            raise SystemExit(f"run {name!r}: no such directory: {path!r}\n"
                             "(an empty path means the shell variable holding it "
                             "was not set in this session)")
    leads = pf.available_leads(runs[0][1])
    meta = pf.read_meta(runs[0][1])

    truth = {}
    for h in leads:
        valid = init + datetime.timedelta(hours=h)
        try:
            n = pf.load_run(runs[0][1], h, meta).sfc.shape[0]
            truth[h] = centre(pf.load_era5(args.era5, args.tc_id, valid, n, meta))
        except FileNotFoundError:
            pass                       # ERA5 is 6-hourly; LLAT steps 3 h
    if not truth:
        raise SystemExit("no ERA5 file matched any forecast hour")

    # The map earns the space it takes. Error magnitude alone cannot distinguish
    # a forecast that lags along the right path from one that goes somewhere else
    # and happens to pass close by, and both occur in the same 240 h run here.
    # Dots at 24 h follow Fig. 4b of the paper, so the two can be read side by side.
    fig, axd = plt.subplot_mosaic([['map', 'tot'], ['map', 'along'], ['map', 'cross']],
                                  figsize=(13.5, 8.0),
                                  gridspec_kw={'width_ratios': [1.25, 1.0]})
    amap = axd['map']
    ax = [axd['tot'], axd['along'], axd['cross']]

    print(f"{'run':<14} {'lead':>5} {'error':>9} {'along':>9} {'cross':>9}")
    print("-" * 52)

    for name, path in runs:
        m = pf.read_meta(path)
        hs = [h for h in pf.available_leads(path) if h in truth]
        fx = [centre(pf.load_run(path, h, m)) for h in hs]

        tot, along, cross = [], [], []
        for h, (flon, flat) in zip(hs, fx):
            tlon, tlat = truth[h]
            ex, ey = km(flon - tlon, flat - tlat, tlat)
            tot.append(np.hypot(ex, ey))
            # Project onto the storm's own direction of travel, taken from truth.
            t0 = truth[min(truth)]
            mx, my = km(tlon - t0[0], tlat - t0[1], tlat)
            norm = np.hypot(mx, my)
            if norm < 1e-6:
                along.append(0.0); cross.append(0.0); continue
            ux, uy = mx / norm, my / norm
            along.append(ex * ux + ey * uy)      # + is ahead of truth
            cross.append(-ex * uy + ey * ux)     # + is left of the track
            print(f"{name:<14} {h:>4}h {tot[-1]:>8.0f} {along[-1]:>+8.0f} "
                  f"{cross[-1]:>+8.0f}  km")

        line, = ax[0].plot(hs, tot, 'o-', lw=1.5, label=name)
        colour = line.get_color()
        ax[1].plot(hs, along, 'o-', lw=1.5, label=name, color=colour)
        ax[2].plot(hs, cross, 'o-', lw=1.5, label=name, color=colour)
        ax[0].set_ylim(bottom=0)

        lons, lats = zip(*fx)
        draw_track(amap, hs, lons, lats, colour, name)
        print(f"{'':<14} track {lons[0]:.2f}E,{lats[0]:.2f}N -> "
              f"{lons[-1]:.2f}E,{lats[-1]:.2f}N")

    th = sorted(truth)
    tl, ta = zip(*[truth[h] for h in th])
    draw_track(amap, th, tl, ta, 'k', 'ERA5', lw=2.2)
    amap.plot(tl[0], ta[0], '*', ms=18, color='k', zorder=5)
    # Longitude spans less ground than latitude away from the equator; without
    # this the track looks like it turns somewhere it does not.
    amap.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(ta))))
    amap.set_title("track — dots every 24 h, star = initial position", fontsize=10)
    amap.set_xlabel("longitude [°E]")
    amap.set_ylabel("latitude [°N]")
    amap.grid(alpha=0.3)
    amap.legend(fontsize=8)
    print(f"{'ERA5':<14} track {tl[0]:.2f}E,{ta[0]:.2f}N -> "
          f"{tl[-1]:.2f}E,{ta[-1]:.2f}N")

    for a, t in zip(ax, ("position error", "along-track (+ = too fast)",
                         "cross-track (+ = left of track)")):
        a.set_title(t, fontsize=10)
        a.set_xlabel("forecast hour")
        a.set_ylabel("km")
        a.axhline(0, c='k', lw=0.8)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{args.tc_id}  init {init:%Y-%m-%d %H}Z", fontsize=12)
    fig.tight_layout()
    fig.savefig(_prepare(args.out), dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--era5", required=True)
    p.add_argument("--tc-id", required=True)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--out", default="track.png")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
