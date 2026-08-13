"""What does the r = 0 ring cost? (B3, measured without training anything.)

`latlon_to_polar` places rings at `linspace(0, r_max, R)`. The first of those has
radius zero, so its 180 azimuthal samples all read the same physical point - a
measured standard deviation of exactly zero along that ring, and 2.4 % of the grid
spent on 179 copies of one number.

The finite-volume alternative puts ring i at (i + 1/2) * r_max / R, so the rings
are cell centres: none is degenerate, they tile the disc evenly, and the innermost
represents the area inside r < dr rather than a single point.

Deciding between them needs no GPU. Both are just resamplings, so send a field
through lat/lon -> polar -> lat/lon each way and compare what comes back. What the
round trip loses is what the parameterisation cannot represent, and training
cannot recover it.

The catch, stated because it is the argument against: cell centres do not sample
r < dr/2 at all, so the exact centre is reconstructed by extrapolation. Whether
that costs more than 179 duplicate samples is exactly what this measures.

Usage
-----
    python tools/rmin_test.py                      # synthetic vortex
    python tools/rmin_test.py --nc path/to.nc --var msl
"""
import argparse

import numpy as np
from scipy.ndimage import map_coordinates


def synthetic(n=81, rmw_deg=0.4, res=0.25):
    """A vortex whose sharpest structure is at the radius of maximum wind.

    An eyewall is the thing at stake, so the test field has one: a pressure
    profile whose gradient peaks near rmw_deg. A smooth blob would round-trip
    perfectly under either scheme and prove nothing.
    """
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    r = np.hypot(xx, yy) * res
    return 101_000.0 - 6_000.0 * np.exp(-(r / rmw_deg) ** 1.4)


def rings(R, r_max_px, cell_centred):
    """Radii of the R rings, in pixels."""
    if cell_centred:
        return (np.arange(R) + 0.5) * r_max_px / R
    return np.linspace(0.0, r_max_px, R)


def round_trip(field, R, Theta, r_max_px, cell_centred):
    """lat/lon -> polar -> lat/lon, returning the reconstruction and the radii."""
    n = field.shape[0]
    c = (n - 1) / 2.0
    r = rings(R, r_max_px, cell_centred)
    th = np.linspace(0.0, 2.0 * np.pi, Theta, endpoint=False)
    rr, tt = np.meshgrid(r, th, indexing='ij')

    # Forward: sample the Cartesian field at each polar node.
    yi = c + rr * np.sin(tt)
    xi = c + rr * np.cos(tt)
    polar = map_coordinates(field, [yi, xi], order=1, mode='nearest')

    # Inverse: for each Cartesian cell, read the polar array at its own (r, theta).
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    rc = np.hypot(xx, yy)
    tc = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)
    # Index into the ring axis: uniform spacing either way, different origin.
    if cell_centred:
        ri = rc * R / r_max_px - 0.5
    else:
        ri = rc * (R - 1) / r_max_px
    ti = tc * Theta / (2.0 * np.pi)
    back = map_coordinates(polar, [np.clip(ri, 0, R - 1), ti],
                           order=1, mode='grid-wrap')
    return np.where(rc <= r_max_px, back, np.nan), rc, polar


def main(args):
    if args.nc:
        import xarray as xr
        with xr.open_dataset(args.nc) as ds:
            field = np.squeeze(ds[args.var].values).astype(float)
            if field.ndim > 2:
                field = field[field.shape[0] // 2]
        ny, nx = field.shape
        n = args.crop or min(ny, nx)
        oy, ox = (ny - n) // 2, (nx - n) // 2
        field = field[oy:oy + n, ox:ox + n]
        crop_note = f"\n  cropped {ny}x{nx} -> {n}x{n}" if (ny, nx) != (n, n) else ""
        label = f"{args.var} from {args.nc}{crop_note}"
    else:
        field = synthetic(args.n, args.rmw, args.res)
        n, label = args.n, f"synthetic vortex, RMW {args.rmw:g} deg"

    r_max_px = (n - 1) / 2.0
    print(f"{label}\n  {n}x{n} cells, R={args.R}, Theta={args.Theta}, "
          f"r_max={r_max_px:g} px = {r_max_px*args.res:g} deg\n")

    print(f"{'scheme':<18} {'innermost ring':>15} {'rms all':>10} "
          f"{'rms r<2deg':>12} {'rms r<RMW':>11}")
    print("-" * 70)
    out = {}
    for name, cc in (("r from 0 (now)", False), ("cell centred (B3)", True)):
        back, rc, polar = round_trip(field, args.R, args.Theta, r_max_px, cc)
        err = np.abs(back - field)
        deg = rc * args.res
        inner = np.nanmean(err[(deg < 2.0)] ** 2) ** 0.5
        core = np.nanmean(err[(deg < args.rmw)] ** 2) ** 0.5
        allr = np.nanmean(err[deg <= r_max_px * args.res] ** 2) ** 0.5
        r0 = rings(args.R, r_max_px, cc)[0] * args.res
        spread = float(np.std(polar[0]))
        print(f"{name:<18} {r0:>8.3f} deg   {'(std %.3g)' % spread:>0} "
              f"{allr:>9.3f} {inner:>11.3f} {core:>10.3f}")
        out[name] = (allr, inner, core)

    a, b = out["r from 0 (now)"], out["cell centred (B3)"]
    print(f"\nchange from the fix, as a fraction of the current error:")
    for i, what in enumerate(("whole disc", "r < 2 deg", "r < RMW")):
        d = (b[i] - a[i]) / a[i] if a[i] else float('nan')
        print(f"  {what:<12} {100*d:+6.1f} %   "
              f"({'better' if d < 0 else 'worse'})")
    print("\nThis bounds what B3 can buy: it is the information the grid keeps or")
    print("throws away, before any model sees it. A small number here means the")
    print("degenerate ring is a tidiness issue and not a forecast one.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nc", default=None, help="a combined.nc to test on instead")
    p.add_argument("--var", default="msl")
    p.add_argument("--crop", type=int, default=81,
                   help="centre-crop the file to this size first. The saved "
                        "combined.nc is 161x161 while the model runs on 81x81, "
                        "and testing the wrong domain gives the wrong dr")
    p.add_argument("--n", type=int, default=81)
    p.add_argument("--R", type=int, default=41)
    p.add_argument("--Theta", type=int, default=180)
    p.add_argument("--res", type=float, default=0.25)
    p.add_argument("--rmw", type=float, default=0.4,
                   help="radius of maximum wind in degrees; the scale that "
                        "matters and the one a 2-degree patch cannot see")
    main(p.parse_args())
