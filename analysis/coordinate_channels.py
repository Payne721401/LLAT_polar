"""How well does the model predict the one thing that sets the track?

The grid follows the storm, so `lon` and `lat` are predicted channels and the
next domain centre is their mean. The forecast track is those two channels and
nothing else - the weather the model produces is never consulted. So the single
number that decides track skill is already in the training logs, and it was never
looked at: it is two of the ninety-eight curves, and the summary chart ranks
variables by RMSE in physical units, where degrees sit far below pascals.

This pulls them out, puts them in kilometres, and sets them against two things
that make them readable: the distance the storm actually covers in one 3 h step,
and the share of the training objective the two channels carry.

Usage
-----
    python analysis/coordinate_channels.py                      # defaults below
    python analysis/coordinate_channels.py --step-km 104        # a fast storm

The per-variable CSV is exported from the training run's TensorBoard logs with
columns tag,step,value; analysis/run_mine_per_var.csv is the one from the
production bf16 / lr 5e-5 run.
"""
import argparse
import collections
import csv
import io
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEG_KM = 111.32

# Both hardcoded in the data pipeline rather than read from the statistics files,
# which is why they have to be repeated here. Changing the grid without changing
# them silently rescales the frame.
LON_STD = 12.0
LAT_STD = 12.0

# From the loss as assembled in models/lightning_modules.py: an unweighted L1 on
# each block, then loss = upper + 0.25 * surface. With no variable weights set in
# config.yaml the criterion is plain nn.L1Loss, so every channel inside a block
# is averaged equally.
SURFACE_WEIGHT = 0.25
N_SURFACE = 20
N_UPPER = 78                                  # 13 pressure levels x 6 variables


def read_tags(path):
    per = collections.defaultdict(dict)
    with io.open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per[r["tag"]][int(r["step"])] = float(r["value"])
    return per


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per = read_tags(args.csv)
    need = ["val_RMSE/lon", "val_RMSE/lat", "val_norm_L1/lon", "val_norm_L1/lat"]
    missing = [t for t in need if t not in per]
    if missing:
        raise SystemExit(f"{args.csv} has no {missing}; export the per-variable "
                         "tags from the run's TensorBoard logs")

    steps = sorted(per["val_RMSE/lon"])
    lon = np.array([per["val_RMSE/lon"][s] for s in steps])
    lat = np.array([per["val_RMSE/lat"][s] for s in steps])
    n_lon = np.array([per["val_norm_L1/lon"][s] for s in steps])
    n_lat = np.array([per["val_norm_L1/lat"][s] for s in steps])

    # Degrees to km. Longitude shrinks with latitude; the validation set is WNP
    # tropical cyclones, so 20 N is the right order for the conversion.
    cos_lat = np.cos(np.deg2rad(args.latitude))
    lon_km, lat_km = lon * DEG_KM * cos_lat, lat * DEG_KM
    pos_km = np.hypot(lon_km, lat_km)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].plot(steps, lon_km, lw=1.4, label="lon")
    axes[0].plot(steps, lat_km, lw=1.4, label="lat")
    axes[0].plot(steps, pos_km, lw=2.0, color='k', label="combined")
    axes[0].axhline(args.step_km, ls='--', c='tab:red', lw=1.4)
    axes[0].annotate(f"one step of storm motion ({args.step_km:.0f} km)",
                     (steps[0], args.step_km * 1.12), color='tab:red',
                     fontsize=8, va='bottom')
    axes[0].set_yscale('log')
    axes[0].set_ylabel("val RMSE [km]")
    axes[0].set_title("single 3 h step: coordinate error against the\n"
                      "distance the storm covers in that step", fontsize=10)

    axes[1].plot(steps, n_lon, lw=1.4, label="lon")
    axes[1].plot(steps, n_lat, lw=1.4, label="lat")
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r"val L1 [$\sigma$]")
    axes[1].set_title("the same error in the units the optimiser sees", fontsize=10)

    # What the objective would gain from a perfect frame. Each surface channel
    # enters the total with weight SURFACE_WEIGHT / N_SURFACE.
    per_channel = SURFACE_WEIGHT / N_SURFACE
    share = per_channel * (n_lon + n_lat)
    if "val_loss" in per:
        vl = np.array([per["val_loss"].get(s, np.nan) for s in steps])
    else:
        vl = np.full_like(share, np.nan)
    axes[2].plot(steps, 100.0 * share / vl if np.isfinite(vl).any()
                 else 100.0 * share / args.val_loss, lw=1.6, color='tab:purple')
    axes[2].set_ylabel("% of the validation objective")
    axes[2].set_title("what the two channels that decide the track\n"
                      "are worth to the loss", fontsize=10)

    for a in axes:
        a.set_xlabel("training step")
        a.grid(alpha=0.3)
    for a in axes[:2]:
        a.legend(fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(HERE, "figures", "fig16_coordinate_channels.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')

    tail = [i for i, s in enumerate(steps) if s > args.converged_after]
    i = tail[-1] if tail else len(steps) - 1
    j = min(tail, key=lambda k: pos_km[k]) if tail else i
    loss = per["val_loss"].get(steps[i], args.val_loss) if "val_loss" in per \
        else args.val_loss

    print(f"final validation state (step {steps[i]}):")
    print(f"  lon  {lon[i]:.3f} deg = {lon_km[i]:6.1f} km   "
          f"({n_lon[i]:.4f} sigma, std {LON_STD:g})")
    print(f"  lat  {lat[i]:.3f} deg = {lat_km[i]:6.1f} km   "
          f"({n_lat[i]:.4f} sigma, std {LAT_STD:g})")
    print(f"  combined position error per 3 h step: {pos_km[i]:.1f} km")
    print(f"  storm motion in one 3 h step:         {args.step_km:.1f} km")
    print(f"  ratio (1.0 means no skill at all):    {pos_km[i]/args.step_km:.2f}")
    print(f"\n  best over the converged tail: {pos_km[j]:.1f} km at step {steps[j]} "
          f"- choosing the checkpoint on this instead of val_loss would gain "
          f"{pos_km[i]-pos_km[j]:.1f} km")
    print(f"\n  share of the objective: {100*per_channel*(n_lon[i]+n_lat[i])/loss:.2f} %"
          f"   (val_loss {loss:.5f})")
    print(f"  one surface channel weighs {per_channel:.5f} against "
          f"{1.0/N_UPPER:.5f} for an upper channel-level")

    print("\n  if the per-step error accumulated as a random walk:")
    for hours in (24, 120, 240):
        n = hours // 3
        print(f"    {hours:>3} h ({n:>2} steps): {pos_km[i]*np.sqrt(n):6.0f} km")
    print("  (the observed Kong-rey one-way errors were 296 km at 24 h and "
          "1048 km at 156 h,\n   so the coordinate channels alone account for "
          "the track error without\n   anything else needing to be wrong)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.path.join(HERE, "run_mine_per_var.csv"))
    p.add_argument("--step-km", type=float, default=104.0,
                   help="distance the storm covers in one 3 h step; the default "
                        "is Kong-rey's observed 833 km/24 h")
    p.add_argument("--latitude", type=float, default=20.0,
                   help="latitude used to convert degrees of longitude to km")
    p.add_argument("--val-loss", type=float, default=0.24997,
                   help="used only if the CSV carries no val_loss tag")
    p.add_argument("--converged-after", type=int, default=60_000,
                   help="ignore steps before this; the run had an instability "
                        "around step 26k that sent these channels to 68 degrees")
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
