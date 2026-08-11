"""How wrong is "the storm does not move"?

The model's single-step coordinate error is 102 km (val_RMSE/lon 0.774 deg,
val_RMSE/lat 0.554 deg, at 20 N). Whether that is good is not answerable on its
own. It needs the score of the most trivial forecast there is - persistence, in
which the domain centre at t+3 h equals the centre at t - because that is what a
model has to beat before it can be said to have learnt anything about storm
motion at all.

Persistence's error is just the distance the storm actually travelled, so the
baseline is the RMS 3-hourly displacement over the same population the model was
scored on.

Two things to keep straight when reading the answer:

  * best-track files are 6-hourly and the model steps 3 h. The training pipeline
    interpolated positions to 3-hourly, and under linear interpolation a 3 h
    displacement is exactly half the 6 h one, so that is what is reported.
  * `val_RMSE/lon` is the RMSE of the longitude *field*, in degrees. A persistence
    forecast gets the whole ramp wrong by one displacement, uniformly, so its
    field RMSE equals its centre error and the two are directly comparable. The
    model's need not be: noise about a correct mean would inflate its RMSE
    without moving its centre, so 102 km is an upper bound on its centre error.

Usage
-----
    python tools/persistence_baseline.py \
        --track-csv /wk2/yungyun/ERA5_2024_for_TC/TC_list_JMA_v2
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

DEG_KM = 111.32

# The paper's own split (its Fig. 8b-d), because it reports track errors about
# 30 % larger for the weakest group and the 202421W case is initialised there.
CLASSES = [("TD  (<35 kt)", 0, 35), ("TS  (35-64 kt)", 35, 65),
           ("TY  (>=65 kt)", 65, 10_000)]

# The current model, for comparison. From analysis/run_mine_per_var.csv at the
# selected checkpoint; see analysis/coordinate_channels.py.
MODEL_LON_DEG, MODEL_LAT_DEG = 0.774, 0.554


def steps(path, hours=6):
    """Consecutive best-track records exactly `hours` apart, as a DataFrame."""
    df = pd.read_csv(path)
    need = {'Year', 'Month', 'Day', 'Hour', 'Lat.', 'Long.'}
    if not need.issubset(df.columns):
        return None
    df['t'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    df = df.sort_values('t').reset_index(drop=True)

    out = df.iloc[:-1].copy()
    out['dt_h'] = (df['t'].shift(-1) - df['t']).dt.total_seconds()[:-1] / 3600.0
    out['dlon'] = df['Long.'].shift(-1)[:-1] - df['Long.'][:-1]
    out['dlat'] = df['Lat.'].shift(-1)[:-1] - df['Lat.'][:-1]
    # Gaps break the assumption that consecutive rows are one interval apart.
    return out[np.isclose(out['dt_h'], hours)]


def report(label, d, per_step):
    """RMS displacement of one group, in degrees and km."""
    if len(d) == 0:
        print(f"  {label:<16} (no samples)")
        return None
    dlon = np.asarray(d['dlon'], dtype=float) * per_step
    dlat = np.asarray(d['dlat'], dtype=float) * per_step
    lat = np.asarray(d['Lat.'], dtype=float)

    rms_lon = float(np.sqrt(np.mean(dlon ** 2)))
    rms_lat = float(np.sqrt(np.mean(dlat ** 2)))
    km_lon = float(np.sqrt(np.mean((dlon * DEG_KM * np.cos(np.deg2rad(lat))) ** 2)))
    km_lat = rms_lat * DEG_KM
    total = float(np.hypot(km_lon, km_lat))
    print(f"  {label:<16} n={len(d):>5}   lon {rms_lon:5.3f}deg {km_lon:6.1f}km"
          f"   lat {rms_lat:5.3f}deg {km_lat:6.1f}km   total {total:6.1f} km")
    return rms_lon, rms_lat, total


def main(args):
    files = sorted(glob.glob(os.path.join(args.track_csv, "*.csv")))
    if not files:
        raise SystemExit(f"no {args.track_csv}/*.csv")

    frames = [s for s in (steps(f, args.record_hours) for f in files)
              if s is not None and len(s)]
    if not frames:
        raise SystemExit("no consecutive best-track records found")
    d = pd.concat(frames, ignore_index=True)

    # 6-hourly records, 3-hourly model steps, linear interpolation between them.
    per_step = args.step_hours / args.record_hours

    print(f"{len(files)} storms, {len(d)} consecutive {args.record_hours} h "
          f"records, scaled to {args.step_hours} h\n")
    print(f"persistence - the error of assuming the storm does not move:")
    overall = report("all", d, per_step)
    if 'Wind (kt)' in d.columns:
        for label, lo, hi in CLASSES:
            report(label, d[(d['Wind (kt)'] >= lo) & (d['Wind (kt)'] < hi)],
                   per_step)

    lon_b, lat_b, total_b = overall
    model_km = float(np.hypot(MODEL_LON_DEG * DEG_KM * np.cos(np.deg2rad(20.0)),
                              MODEL_LAT_DEG * DEG_KM))
    print(f"\nthe model, single {args.step_hours} h step:"
          f"        lon {MODEL_LON_DEG:5.3f}deg   lat {MODEL_LAT_DEG:5.3f}deg"
          f"   total {model_km:6.1f} km")
    print(f"persistence, same units:              "
          f"lon {lon_b:5.3f}deg   lat {lat_b:5.3f}deg   total {total_b:6.1f} km")
    skill = 1.0 - model_km / total_b
    print(f"\nskill against persistence: {100*skill:+.0f} %"
          f"   ({'the model beats it' if skill > 0 else 'DOING NOTHING SCORES BETTER'})")
    print("\ncaveats: these are the years in this CSV directory, not the "
          "validation years;\n  and the model was trained on JTWC-centred "
          "domains while this is JMA.\n  Translation speed statistics are stable "
          "enough between agencies and years\n  that the comparison survives "
          "both, but the number is indicative, not exact.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track-csv", required=True,
                   help="directory of {TC_ID}.csv best-track files")
    p.add_argument("--record-hours", type=float, default=6.0,
                   help="spacing of the best-track records")
    p.add_argument("--step-hours", type=float, default=3.0,
                   help="the model's step; the displacement is scaled to it")
    main(p.parse_args())
