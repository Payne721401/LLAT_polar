"""Which surface forcings follow the storm, and which one does not?

202421W from 2024-10-27 00Z gives P1 a 940 hPa cyclone at 49 N in late October,
crossing Honshu at +153 h without weakening. Neither is possible: the sea there
is 10-15 C against the 26 C a tropical cyclone needs, and a landfall over 400 m
of terrain should tear the core apart. The forecast has no mechanism to decay
because it is not being told to.

The reason is one line in the code. Every step, changing_additional_information
calls recalc_additional_np, which recomputes the fields that depend on where the
storm now is and what time it now is, and overwrites the model's predictions of
them:

    landmask  hgt  f  solar  diurnal_sin  diurnal_cos  doy_sin  doy_cos

sst_filled is not in that list. It is the one surface forcing the model predicts
and then eats, for the whole rollout. Over a 3-hour training step "the sea
surface temperature next step is the sea surface temperature now" is very nearly
the optimal prediction, so that is what gets learned; over 192 hours and 33
degrees of latitude it means the storm carries its tropical ocean north with it.

The internal control is what makes this decisive, and it needs no external truth
at all. landmask and hgt DO track the position, sst_filled does not, and they are
the same model at the same step. If the mask jumps to 1 over Japan while the sea
underneath stays tropical, the mechanism is demonstrated rather than argued.

Usage
-----
    B=$HOME/LLAT_polar_runs
    P=$HOME/LLAT_polar_runs_p1
    python tools/static_channel_drift.py \\
        --run "baseline=$B/202421W/one_way_couple_model_LLAT_polar_vtvr_v1/start_from_2024102700" \\
        --run "P1=$P/202421W/one_way_couple_model_LLAT_polar_p1_v1/start_from_2024102700" \\
        --tc-id 202421W --init 2024102700
"""
import argparse
import datetime
import importlib.util
import os
import warnings

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "plot_forecast", os.path.join(_here, "plot_forecast.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

# Recomputed from position and time every step by recalc_additional_np, so these
# are prescribed rather than predicted whatever the model outputs for them.
PRESCRIBED = ('landmask', 'hgt', 'f', 'solar')
# Predicted, fed back, and never corrected.
FREE = ('sst_filled',)

KELVIN = 273.15
TC_THRESHOLD = 26.5     # deg C, the conventional floor for maintenance


def core_mean(field, name, radius_deg=1.0):
    """A channel averaged over the inner core, not sampled at one point.

    A single central cell is noisy and, for sst_filled, sits under the eye where
    the model has every reason to have done something unusual. One degree is
    about the eyewall and its surroundings.
    """
    a = np.asarray(field.s(name), dtype=float)
    n = a.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    m = (np.hypot(xx, yy) * res <= radius_deg) & np.isfinite(a)
    return float(a[m].mean()) if m.any() else np.nan


def main(args):
    warnings.filterwarnings("ignore", message="All-NaN slice")
    init = datetime.datetime.strptime(args.init, "%Y%m%d%H")

    for spec in args.run:
        label, _, path = spec.partition('=')
        if not path:
            label, path = os.path.basename(spec.rstrip('/')), spec
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            raise SystemExit(f"no such directory: {path}\nA shell variable in "
                             f"--run may be unset.")
        meta = pf.read_meta(path)
        leads = pf.available_leads(path)
        if not leads:
            raise SystemExit(f"{path} holds no output_sfc_*h.npy")
        if args.max_lead is not None:
            leads = [h for h in leads if h <= args.max_lead]

        print(f"\n{label}   {args.tc_id} init {init:%Y-%m-%d %HZ}")
        print(f"{'lead':>6}{'lat':>8}{'lon':>9}"
              f"{'SST':>9}{'t2m':>8}{'landmask':>10}{'hgt':>8}{'f':>10}")
        print(f"{'':>6}{'':>8}{'':>9}{'[C]':>9}{'[C]':>8}"
              f"{'(fixed)':>10}{'(fixed)':>8}{'(fixed)':>10}")
        print("-" * 68)

        rows = []
        for h in leads[::args.every]:
            f = pf.load_run(path, h, meta)
            lat = float(np.nanmean(f.lat))
            sst = core_mean(f, 'sst_filled', args.radius)
            row = dict(
                h=h, lat=lat, lon=float(np.nanmean(f.lon)),
                sst=sst - KELVIN if np.isfinite(sst) and sst > 100 else sst,
                t2m=core_mean(f, 't2m', args.radius) - KELVIN,
                lm=core_mean(f, 'landmask', args.radius),
                hgt=core_mean(f, 'hgt', args.radius),
                cor=core_mean(f, 'f', args.radius),
            )
            rows.append(row)
            print(f"{row['h']:>6.0f}{row['lat']:>8.2f}{row['lon']:>9.2f}"
                  f"{row['sst']:>9.1f}{row['t2m']:>8.1f}{row['lm']:>10.2f}"
                  f"{row['hgt']:>8.0f}{row['cor']:>10.2e}")

        if len(rows) < 2:
            continue
        a, b = rows[0], rows[-1]
        dlat = b['lat'] - a['lat']
        print()
        print(f"  latitude   {a['lat']:>7.2f} -> {b['lat']:>7.2f}"
              f"   ({dlat:+.2f} deg)")
        print(f"  SST        {a['sst']:>7.1f} -> {b['sst']:>7.1f} C"
              f"   ({b['sst'] - a['sst']:+.1f})   PREDICTED, never corrected")
        print(f"  Coriolis   {a['cor']:>7.2e} -> {b['cor']:>7.2e}"
              f"   prescribed from latitude every step")

        # The internal control. f is a pure function of latitude, so it says how
        # far the frame really moved; SST should have followed it and did not.
        if abs(dlat) > 5.0:
            expected = -0.45 * dlat        # rough late-October western Pacific
            actual = b['sst'] - a['sst']
            print(f"\n  Moving {dlat:+.1f} degrees of latitude in late October")
            print(f"  should cool the sea underneath by roughly"
                  f" {expected:.0f} C. It changed by {actual:+.1f}.")
            if abs(actual) < 0.4 * abs(expected):
                print("  The sea did not follow the storm. Coriolis did, and the")
                print("  land mask did, because those are recomputed from the new")
                print("  position every step and sst_filled is not - it is the")
                print("  model's own prediction fed back for the whole rollout.")
                print("  That is why nothing forces the storm to decay.")

        warm = [r for r in rows if np.isfinite(r['sst'])
                and r['sst'] > TC_THRESHOLD]
        if warm and abs(warm[-1]['lat']) > 35:
            print(f"\n  Still over {TC_THRESHOLD} C water at"
                  f" {warm[-1]['lat']:.1f} N at +{warm[-1]['h']:.0f} h."
                  f" There is no such water at that latitude in October.")

        land = [r for r in rows if r['lm'] > 0.5]
        if land:
            print(f"  Meanwhile the land mask reaches"
                  f" {max(r['lm'] for r in land):.2f} at"
                  f" +{land[0]['h']:.0f} h, so the position IS getting through -"
                  f" to every forcing except the one that governs the heat.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="label=path, repeatable")
    p.add_argument("--tc-id", required=True)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--radius", type=float, default=1.0,
                   help="degrees averaged over, around the centre")
    p.add_argument("--every", type=int, default=4,
                   help="print every Nth lead; 4 is 12-hourly at a 3 h step")
    p.add_argument("--max-lead", type=float, default=None)
    main(p.parse_args())
