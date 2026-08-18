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
import os

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

    if args.map:
        back, rc, _ = round_trip(field, args.R, args.Theta, r_max_px, False)
        error_map(field, back, rc, args, label)


def axisymmetry(err, deg, nbins=40):
    """How much of the error is a function of radius alone?

    The distinction that decides where an artefact comes from. Resampling error
    that is unstructured shows up as speckle and averages away around a ring;
    error produced BY the ring geometry - a grid too coarse at one radius, the
    degenerate centre, the join between rings - is the same at every azimuth and
    survives the azimuthal mean. Concentric rings in a plotted vorticity field
    are the second kind, and this separates them without anyone squinting at a
    picture.

    Returns the bin centres, the azimuthal-mean profile, and the fraction of the
    error variance that profile carries. A high fraction means the artefact
    belongs to the grid, so no amount of training removes it and the sampling has
    to change. A low one means the model is producing it.

    READ THIS ON A REAL FIELD, NOT THE SYNTHETIC VORTEX. The synthetic field is
    axisymmetric by construction, so its error has no choice but to be, and it
    scores about 90 % whatever the transform does. On an ERA5 field, which has
    fronts and asymmetric rainbands, a high score is evidence rather than an
    artefact of the test.
    """
    m = np.isfinite(err)
    edges = np.linspace(0, float(np.nanmax(deg[m])), nbins + 1)
    idx = np.clip(np.digitize(deg, edges) - 1, 0, nbins - 1)
    prof = np.array([np.nanmean(err[m & (idx == i)]) if (m & (idx == i)).any()
                     else np.nan for i in range(nbins)])
    sym = prof[idx]
    tot = float(np.nanvar(err[m]))
    frac = float(np.nanvar(sym[m]) / tot) if tot > 0 else float('nan')
    return 0.5 * (edges[:-1] + edges[1:]), prof, frac


def error_map(field, back, rc, args, label):
    """Draw what the round trip did, and say which kind of error it is."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    err = back - field
    deg = rc * args.res
    centres, prof, frac = axisymmetry(np.abs(err), deg)

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.2))
    lim = float(np.nanmax(deg))
    ext = [-lim, lim, -lim, lim]
    v = float(np.nanpercentile(np.abs(err[np.isfinite(err)]), 99))
    for a, (arr, ttl, kw) in zip(ax, [
            (field, "original", dict(cmap='viridis')),
            (back, "after polar round trip", dict(cmap='viridis')),
            (err, "difference", dict(cmap='RdBu_r', vmin=-v, vmax=v))]):
        im = a.imshow(arr, origin='lower', extent=ext, **kw)
        a.set_title(ttl)
        a.set_xlabel("deg")
        fig.colorbar(im, ax=a, fraction=0.046)

    ax[3].plot(centres, prof, '-o', ms=3)
    ax[3].set_xlabel("radius [deg]")
    ax[3].set_ylabel("azimuthal mean |error|")
    ax[3].set_title(f"axisymmetric fraction {100 * frac:.0f}%")
    ax[3].grid(alpha=0.3)

    fig.suptitle(f"{label}   R={args.R}  Theta={args.Theta}")
    out = os.path.expanduser(args.map)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)

    print(f"\nwrote {out}")
    print(f"  {100 * frac:.0f}% of the round-trip error variance is a function")
    print(f"  of radius alone.", end=" ")
    if frac > 0.5:
        print("That is a ring artefact: the same at every")
        print("  azimuth because the grid produced it, not the field. Training")
        print("  cannot remove it; the sampling has to change.")
    elif frac > 0.2:
        print("Part ring, part speckle - the rings are")
        print("  real but they are not the whole story.")
    else:
        print("Mostly unstructured. The concentric")
        print("  rings in the forecast fields are NOT coming from this transform;")
        print("  look at what the model produces in polar space instead.")


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
    p.add_argument("--map", default=None, metavar="PNG",
                   help="also draw the round-trip error field and split it into "
                        "the part that is a function of radius alone - which is "
                        "what a concentric-ring artefact is - and the rest")
    p.add_argument("--rmw", type=float, default=0.4,
                   help="radius of maximum wind in degrees; the scale that "
                        "matters and the one a 2-degree patch cannot see")
    main(p.parse_args())
