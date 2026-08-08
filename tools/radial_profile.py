"""Azimuthal statistics against radius: where does structure stop being physical?

Two very different things degrade a polar forecast near the rim, and on a map
they look alike - wiggly isobars, patchy wind:

  1. **The boundary seam.** Whatever supplies the lateral boundary - a frozen
     initial condition in standalone mode, FCNV2 when coupled - is discontinuous
     with the evolving interior at the radius where it takes over. That shows up
     as a *step* at exactly that radius.
  2. **The outer-ring artefact.** An unweighted loss gives every ring the same
     number of points while ring area grows with r, so per-unit-area weight falls
     as 1/r and the outer domain is the least constrained part of the grid. That
     shows up as a *gradual* rise in azimuthal variability toward r_max, steepest
     in the outermost ring or two.

A radial profile separates them: a step moves when the boundary radius moves, a
gradient does not. Run the same forecast twice with different --hold-radius and
compare, or compare standalone against one-way.

Usage
-----
    python tools/radial_profile.py \
        --run "hold 9=~/runs/.../start_from_2024102500" \
        --run "hold 7=~/runs_h7/.../start_from_2024102500" \
        --lead 24 --out radial.png
"""
import argparse
import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "plot_forecast", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "plot_forecast.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

# Fields whose azimuthal roughness is diagnostic. Pressure is the most sensitive:
# it is smooth in nature, so any small-scale azimuthal structure is suspect.
FIELDS = [
    ("MSLP", lambda f: f.s('msl') / 100.0, "hPa"),
    ("10 m wind", lambda f: np.hypot(f.s('u10'), f.s('v10')), "m s$^{-1}$"),
    ("850 hPa vorticity", lambda f: f.vorticity(850) * 1e5, "10$^{-5}$ s$^{-1}$"),
]


def radial_bins(field, n_bins=40):
    """Radius of each cell in degrees, and bin edges out to the domain corner."""
    n = field.lon.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    r = np.hypot(xx, yy) * res
    return r, np.linspace(0, r.max(), n_bins + 1)


def profile(field, get):
    """Azimuthal mean per radial bin, and the spread about the axisymmetric part.

    The spread is the diagnostic. A TC is close to axisymmetric, so departures
    should stay flat or fall with radius; a rise toward the rim is structure that
    is not the storm.

    The subtlety is that bins have width, so a plain within-bin standard
    deviation also counts the *radial* gradient across the bin. Near the eyewall
    that gradient is enormous - order 10 hPa per degree - and would swamp the
    azimuthal signal exactly where the storm is. So the axisymmetric component is
    removed first: build the mean profile, interpolate it back to each cell's own
    radius, and take the spread of the residual. What remains is asymmetry alone.
    """
    z = np.asarray(get(field), dtype=float)
    r, edges = radial_bins(field)
    ok = np.isfinite(z)

    mid, mean = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & ok
        if m.sum() < 8:                      # too few points to say anything
            continue
        mid.append(0.5 * (lo + hi))
        mean.append(np.mean(z[m]))
    mid, mean = np.array(mid), np.array(mean)
    if len(mid) < 2:
        return mid, mean, np.zeros_like(mean)

    resid = z - np.interp(r, mid, mean)
    std = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & ok
        if m.sum() < 8:
            continue
        std.append(np.std(resid[m]))
    return mid, mean, np.array(std)


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [r.split('=', 1) for r in args.run]
    cols = [(name, pf.load_run(path, args.lead, pf.read_meta(path)))
            for name, path in runs]

    fig, axes = plt.subplots(2, len(FIELDS), figsize=(4.2 * len(FIELDS), 6.4),
                             squeeze=False, sharex=True)
    for c, (label, get, unit) in enumerate(FIELDS):
        for name, f in cols:
            mid, mean, std = profile(f, get)
            axes[0][c].plot(mid, mean, lw=1.4, label=name)
            axes[1][c].plot(mid, std, lw=1.4, label=name)
        axes[0][c].set_title(label, fontsize=10)
        axes[0][c].set_ylabel(f"azimuthal mean [{unit}]", fontsize=8)
        axes[1][c].set_ylabel(f"azimuthal std [{unit}]", fontsize=8)
        axes[1][c].set_xlabel("radius [deg]")
        for a in (axes[0][c], axes[1][c]):
            for rad in args.mark:
                a.axvline(rad, ls=':', c='0.5', lw=1)
            a.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)

    fig.suptitle(f"+{args.lead:03d} h — dotted lines: "
                 + ", ".join(f"r = {m:g}°" for m in args.mark), fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"wrote {args.out}")

    print("\nazimuthal std of MSLP by radius (hPa) — a step marks the boundary "
          "seam, a rise marks the outer-ring artefact")
    for name, f in cols:
        mid, _, std = profile(f, FIELDS[0][1])
        pick = [np.argmin(abs(mid - x)) for x in (2, 5, 7, 8, 9, 9.5)]
        print(f"  {name:<20} " + "  ".join(
            f"r={mid[i]:>4.1f}:{std[i]:>6.2f}" for i in pick))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--lead", type=int, required=True)
    p.add_argument("--out", default="radial.png")
    p.add_argument("--mark", type=float, nargs="*", default=[8.0, 9.0, 10.0],
                   help="radii to mark, in degrees; put your --hold-radius here")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
