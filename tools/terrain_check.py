"""Did the forecast storm ever go near land, and did the truth?

WRITTEN BECAUSE THE QUESTION WAS ANSWERED WITHOUT BEING MEASURED. Asked whether
202421W's lack of terrain-induced weakening meant the model was missing the
terrain, the answer given was that the storm never approaches Taiwan - inferred
from the last position of the track and nothing else. A track that ends east of
land can still have passed near land, or near the Ryukyus, or over a small
island, and an endpoint says nothing about any of that.

So this measures it, from the model's own landmask channel rather than from a
coastline drawn somewhere else. That matters: what decides whether the forecast
could have weakened over terrain is the land the MODEL was shown, not the land
that exists.

Reports, for every lead:

  - the centre position, and the landmask and surface height AT the centre
  - the largest landmask and height anywhere within --core-deg of the centre,
    which is what a storm actually feels: a circulation of 200 km radius
    interacts with a coast long before its centre crosses one
  - the distance from the centre to the nearest land point in the domain

and draws the tracks over the land the model saw, so the numbers have a picture
attached. Truth is drawn too when --era5 is given, because the interesting
failure is a forecast that stays at sea while the real storm makes landfall.

Usage
-----
    R=$HOME/scratch/season/p1
    python tools/terrain_check.py \\
        --run "P1=$R/202421W/one_way_couple_model_LLAT_polar_p1_v1/start_from_2024102600" \\
        --era5 /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty \\
        --tc-id 202421W --init 2024102600

The figure lands in analysis/figures/forecasts/<TCID>/<init>/terrain.png unless
--out says otherwise.
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "plot_forecast", os.path.join(_here, "plot_forecast.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

DEG_KM = 111.32
LAND = 0.5          # landmask above this counts as land
SEA_LEVEL = 50.0    # metres; above this counts as terrain worth naming


def centre(field):
    """The frame centre, which is the storm centre by construction.

    The coordinate channels are predicted and their mean is what the model uses
    to move its own frame, so this is the model's own idea of where the storm
    is - not a re-detection of it.
    """
    return float(np.nanmean(field.lon)), float(np.nanmean(field.lat))


def km_between(lon0, lat0, lon, lat):
    """Distance in km on a local flat-earth approximation.

    Good to a fraction of a percent over the ten degrees this domain spans, and
    avoids importing a geodesy package for a number that is compared against
    thresholds of a hundred kilometres.
    """
    dx = (lon - lon0) * DEG_KM * np.cos(np.deg2rad(lat0))
    dy = (lat - lat0) * DEG_KM
    return np.hypot(dx, dy)


def probe(field, core_deg):
    """Land and terrain at the centre, near the centre, and nearest to it."""
    lm = np.asarray(field.s('landmask'), dtype=float)
    try:
        hgt = np.asarray(field.s('hgt'), dtype=float)
    except ValueError:
        hgt = np.full_like(lm, np.nan)
    lon, lat = np.asarray(field.lon, float), np.asarray(field.lat, float)
    clon, clat = centre(field)

    n = lm.shape[0]
    i = j = (n - 1) // 2
    d = km_between(clon, clat, lon, lat)

    near = (d <= core_deg * DEG_KM) & np.isfinite(lm)
    land = (lm > LAND) & np.isfinite(lm)

    return dict(
        lon=clon, lat=clat,
        lm_centre=float(lm[i, j]),
        hgt_centre=float(hgt[i, j]),
        lm_near=float(np.nanmax(lm[near])) if near.any() else np.nan,
        hgt_near=float(np.nanmax(hgt[near])) if near.any() else np.nan,
        d_land=float(np.nanmin(d[land])) if land.any() else np.inf,
        land_lon=lon[land], land_lat=lat[land],
    )


def truth_track(era5_dir, tc_id, init, leads, n, meta, core_deg):
    """Where the storm really was, from the analysed TC-centred files."""
    out = {}
    for h in leads:
        t = init + datetime.timedelta(hours=int(h))
        try:
            f = pf.load_era5(era5_dir, tc_id, t, n, meta)
        except (FileNotFoundError, OSError):
            continue
        out[h] = probe(f, core_deg)
    return out


def default_out(tc_id, init):
    """analysis/figures/forecasts/<TCID>/<init>/terrain.png

    The layout the other figures already use - one directory per case and per
    initial time, with the filename saying which plot it is - so a case's
    figures stay together instead of by figure type.
    """
    return os.path.join("analysis", "figures", "forecasts", tc_id, init,
                        "terrain.png")


def draw(ax, hours, lons, lats, colour, label):
    ax.plot(lons, lats, '-', color=colour, lw=1.8, label=label, zorder=3)
    step = max(1, int(round(24.0 / max(1, hours[1] - hours[0])))) if len(hours) > 1 else 1
    ax.plot(lons[::step], lats[::step], 'o', color=colour, ms=4.5, zorder=4)
    for h, x, y in list(zip(hours, lons, lats))[::step][1:]:
        ax.annotate(f"{h:.0f}", (x, y), textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color=colour, zorder=5)


def main(args):
    init = datetime.datetime.strptime(args.init, "%Y%m%d%H")
    runs = []
    for spec in args.run:
        label, _, path = spec.partition('=')
        if not path:
            label, path = os.path.basename(spec.rstrip('/')), spec
        runs.append((label, os.path.expanduser(path)))

    colours = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    series, land_pts = {}, []

    for label, path in runs:
        # An unset shell variable in --run makes an absolute path that begins
        # at the root, and every step downstream degrades quietly: read_meta
        # only warns, available_leads returns nothing, the table prints its
        # header and no rows, and the first thing to actually raise is a min()
        # over an empty sequence six frames away. Say it here instead.
        if not os.path.isdir(path):
            raise SystemExit(
                f"no such directory: {path}\n"
                f"If that begins with a stray '/', a shell variable in --run was "
                f"unset - `R=$HOME/scratch/season/p1` has to be set in the same "
                f"shell as the command that uses it.")
        meta = pf.read_meta(path)
        leads = pf.available_leads(path)
        if args.max_lead is not None:
            leads = [h for h in leads if h <= args.max_lead]
        if not leads:
            raise SystemExit(
                f"{path} exists but holds no output_sfc_*h.npy under "
                f"{os.path.relpath(pf.forecast_dir(path), path)}. Either the "
                f"forecast has not been run, or this is the wrong level of the "
                f"directory tree.")
        rows = {}
        for h in leads:
            f = pf.load_run(path, h, meta)
            rows[h] = probe(f, args.core_deg)
            land_pts.append((rows[h]['land_lon'], rows[h]['land_lat']))
        series[label] = rows

    if args.era5:
        label = 'ERA5'
        any_run = runs[0][1]
        meta = pf.read_meta(any_run)
        leads = sorted(next(iter(series.values())))
        rows = truth_track(os.path.expanduser(args.era5), args.tc_id, init,
                           leads, args.n, meta, args.core_deg)
        if rows:
            series[label] = rows
            for r in rows.values():
                land_pts.append((r['land_lon'], r['land_lat']))

    # ── the table, which is the actual answer ───────────────────────────
    for label, rows in series.items():
        print(f"\n{label}   {args.tc_id} init {init:%Y-%m-%d %HZ}")
        print(f"{'lead':>6}{'lon':>9}{'lat':>8}"
              f"{'land@c':>9}{'hgt@c':>8}"
              f"{'land<' + format(args.core_deg, 'g') + 'd':>10}"
              f"{'hgt near':>10}{'km to land':>12}")
        print("-" * 72)
        for h in sorted(rows):
            r = rows[h]
            dl = "  (none in domain)" if not np.isfinite(r['d_land']) \
                else f"{r['d_land']:>12.0f}"
            print(f"{h:>6.0f}{r['lon']:>9.2f}{r['lat']:>8.2f}"
                  f"{r['lm_centre']:>9.2f}{r['hgt_centre']:>8.0f}"
                  f"{r['lm_near']:>10.2f}{r['hgt_near']:>10.0f}{dl}")

        over = [h for h in sorted(rows) if rows[h]['lm_centre'] > LAND]
        near = [h for h in sorted(rows) if rows[h]['d_land'] <= args.near_km]
        closest = min(rows, key=lambda h: rows[h]['d_land'])
        print()
        if over:
            print(f"  centre over land at +{over[0]:.0f} h"
                  f" ({len(over)} lead(s)); highest terrain within"
                  f" {args.core_deg:g} deg was"
                  f" {max(rows[h]['hgt_near'] for h in over):.0f} m")
        elif near:
            print(f"  never over land, but within {args.near_km:.0f} km of it"
                  f" at +{near[0]:.0f} h; closest approach"
                  f" {rows[closest]['d_land']:.0f} km at +{closest:.0f} h")
        elif not np.isfinite(rows[closest]['d_land']):
            print("  no land in the domain at ANY lead. The storm is not merely")
            print("  offshore, it is more than a domain radius from any coast.")
        else:
            print(f"  never within {args.near_km:.0f} km of land."
                  f" Closest approach {rows[closest]['d_land']:.0f} km"
                  f" at +{closest:.0f} h.")
            print("  A storm this far offshore cannot weaken over terrain, so")
            print("  the absence of terrain weakening says nothing about whether")
            print("  the model represents terrain. Test that on a case that")
            print("  makes landfall in the truth.")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # `land_pts` is non-empty as soon as any file was read, but every entry can
    # still be an empty array - which is exactly what happens for the case this
    # tool was written for, a storm that never comes near land. Filtering has to
    # come before the truth test, or np.concatenate gets an empty list and
    # raises, and the figure fails precisely when the answer is most decisive.
    have = [p for p in land_pts if p[0].size]
    if have:
        lo = np.concatenate([p[0] for p in have])
        la = np.concatenate([p[1] for p in have])
        # Deduplicate on a coarse grid: every lead contributes its own copy of
        # the same coastline, and scattering all of them is slow and opaque.
        key = np.unique(np.round(np.stack([lo, la]) / 0.25).astype(int), axis=1)
        ax.scatter(key[0] * 0.25, key[1] * 0.25, s=6, c='0.72', marker='s',
                   linewidths=0, zorder=1,
                   label='land, as the model sees it')
    else:
        ax.text(0.5, 0.02, "no land anywhere in any domain, at any lead",
                transform=ax.transAxes, ha='center', fontsize=9, color='0.4')

    for (label, rows), colour in zip(series.items(), colours):
        hours = sorted(rows)
        draw(ax, [float(h) for h in hours],
             np.array([rows[h]['lon'] for h in hours]),
             np.array([rows[h]['lat'] for h in hours]),
             'k' if label == 'ERA5' else colour, label)

    lats = [r['lat'] for rows in series.values() for r in rows.values()]
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(lats))))
    ax.grid(alpha=0.3)
    ax.set_xlabel('longitude'), ax.set_ylabel('latitude')
    ax.legend(loc='best', fontsize=9)
    ax.set_title(f"{args.tc_id}  init {init:%Y-%m-%d %HZ}  "
                 f"track against the land the model was shown")

    out = os.path.expanduser(args.out or default_out(args.tc_id, args.init))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="label=path, repeatable")
    p.add_argument("--era5", help="TC-centred ERA5 directory, for the truth track")
    p.add_argument("--tc-id", required=True)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--out", default=None,
                   help="defaults to analysis/figures/forecasts/<TCID>/<init>/"
                        "terrain.png, alongside the other figures for the case")
    p.add_argument("--core-deg", type=float, default=2.0,
                   help="radius around the centre searched for land and terrain; "
                        "2 deg is about the outer eyewall and inner rainbands")
    p.add_argument("--near-km", type=float, default=200.0,
                   help="how close counts as a land interaction")
    p.add_argument("--max-lead", type=float, default=None,
                   help="stop here; beyond about +72 h the forecast and the "
                        "truth are in different places and comparing what they "
                        "sit over stops meaning anything")
    p.add_argument("--n", type=int, default=81,
                   help="ERA5 grid size to read, to match the forecast domain")
    p.add_argument("--no-plot", action="store_true")
    main(p.parse_args())
