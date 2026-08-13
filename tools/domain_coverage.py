"""Is a 10-degree disc big enough to know which way the storm goes?

The three worst cases of the 2024 season fail the same way, and it is not the
speed bias. 202414W and 202422W all turn the forecast storm north while the real
one keeps going west: cross-track reaches 2058, 1666 and 1482 km against
along-track of 797, 1265 and 1327. A storm that recurves when it should not is a
steering failure, not a lagging one.

Steering comes from the subtropical ridge, which for a storm at 11-16 N sits
around 25-30 N - ten to nineteen degrees away, at or beyond the edge of what the
model can see. And the polar grid sees less than the Cartesian one it replaced:
a disc of radius 10 against a square reaching 14.1 degrees into its corners, so
the diagonals, where a ridge to the north-east would be, are exactly what was
given up.

The test is which radius best predicts the storm's actual motion, not which
radius the disc mean stops changing at.

That distinction cost a wrong conclusion once and is worth stating. A first
version reported only how much the disc-mean shifts between 8 and 14 degrees, got
3.25 m/s, and declared the domain too small. It does not follow: the large-scale
flow varies with radius for every storm, and a 14-degree mean sweeps in a great
deal that does not steer anything. The literature settles on a deep-layer mean
over roughly 5-7 degrees precisely because that is what predicts motion best -
which would make 10 degrees ample.

So the question has to be asked against the answer. For each radius, compare the
steering there with where the storm actually went. Whichever radius tracks the
motion most closely is the one that matters, and only if that lies outside the
domain is the boundary implicated.

No model and no training involved - this is a property of the atmosphere and of
where the domain boundary was drawn.

Usage
-----
    python tools/domain_coverage.py \\
        --era5 /wk2/yungyun/FCNV2_TC/202414W/ERA5/for_DLAMPty \\
        --tc-id 202414W --times 2024091600,2024091612,2024091700
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "steering", os.path.join(_here, "steering.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)
pf = st.pf

DEG_KM = 111.32


def steering_at(field, radius_deg, levels):
    """Deep-layer mean wind over a disc of this radius, in m/s."""
    n = field.lon.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    mask = np.hypot(xx, yy) * res <= radius_deg
    if mask.sum() < 20:
        return None
    uv = []
    for L in st.deep_layer(levels):
        u = np.asarray(field.u('u', L), dtype=float)[mask]
        v = np.asarray(field.u('v', L), dtype=float)[mask]
        uv.append((np.nanmean(u), np.nanmean(v)))
    uv = np.array(uv)
    return float(uv[:, 0].mean()), float(uv[:, 1].mean())


def main(args):
    era5 = os.path.expanduser(args.era5)
    times = [datetime.datetime.strptime(t, "%Y%m%d%H")
             for t in args.times.split(',')]
    radii = [float(r) for r in args.radii.split(',')]

    # Where the storm actually went, from the analysed centres either side.
    centres = {}
    for t in times:
        for dt in (-args.dt, 0, args.dt):
            tt = t + datetime.timedelta(hours=dt)
            if tt in centres:
                continue
            try:
                g = pf.load_era5(era5, args.tc_id, tt, args.n, None)
            except FileNotFoundError:
                continue
            centres[tt] = (float(np.nanmean(g.lon)), float(np.nanmean(g.lat)))

    print("steering at each radius, against where the storm actually went\n")
    print(f"{'valid':<16}{'motion':>9}" +
          "".join(f"{r:>6.0f}deg" for r in radii))
    print("-" * (25 + 9 * len(radii)))

    err = {r: [] for r in radii}
    for t in times:
        a = centres.get(t - datetime.timedelta(hours=args.dt))
        b = centres.get(t + datetime.timedelta(hours=args.dt))
        if not (a and b):
            print(f"{t:%Y-%m-%d %HZ}   (no motion: needs +/-{args.dt} h)")
            continue
        lat = centres[t][1] if t in centres else b[1]
        dx, dy = st.km(b[0] - a[0], b[1] - a[1], lat)
        dt_s = 2 * args.dt * 3600.0
        mot = (dx * 1000.0 / dt_s, dy * 1000.0 / dt_s)

        try:
            f = pf.load_era5(era5, args.tc_id, t, args.n, None)
        except FileNotFoundError:
            continue
        cells = ""
        for r in radii:
            v = steering_at(f, r, pf.LEVELS)
            if v is None:
                cells += f"{'-':>9}"
                continue
            e = float(np.hypot(v[0] - mot[0], v[1] - mot[1]))
            err[r].append(e)
            cells += f"{e:>9.2f}"
        print(f"{t:%Y-%m-%d %HZ} {np.hypot(*mot):>8.1f} {cells}")

    have = [r for r in radii if err[r]]
    if have:
        med = {r: float(np.median(err[r])) for r in have}
        best = min(med, key=med.get)
        print(f"\n{'radius':>8} {'median |steering - motion|':>28}")
        for r in have:
            mark = "  <- best" if r == best else ""
            print(f"{r:>6.0f}deg {med[r]:>24.2f} m/s{mark}")
        print(f"\nThe steering that best matches the motion is at {best:g} degrees.")
        if best >= max(have) - 1e-9:
            print("That is the largest radius tested, so the relationship has not")
            print("turned over and a wider domain might still be buying something;")
            print("extend --radii before concluding anything.")
        elif best > 10.0:
            print("That is OUTSIDE the 10-degree disc the model sees, so the")
            print("boundary is implicated and no amount of training fixes it.")
        else:
            print("That is INSIDE the 10-degree disc, so the model is shown the")
            print("flow that matters and these failures are not about the domain")
            print("size. Look at what it does with it instead - tools/steering.py")
            print("compares the model's own steering against ERA5's.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era5", required=True)
    p.add_argument("--tc-id", required=True)
    p.add_argument("--times", required=True,
                   help="comma-separated YYYYMMDDHH")
    p.add_argument("--radii", default="2,4,6,8,10,12,14",
                   help="disc radii in degrees; beyond 10 needs the uncropped "
                        "161x161 file, which is what for_DLAMPty holds")
    p.add_argument("--dt", type=int, default=6,
                   help="half-window for the centred difference that gives the "
                        "observed motion; 6 h means comparing t-6 with t+6")
    p.add_argument("--n", type=int, default=161,
                   help="grid size to read; 161 keeps the corners the model "
                        "does not get to see")
    main(p.parse_args())
