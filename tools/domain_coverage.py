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

This measures the size of that. For a real analysis it computes the deep-layer
steering over discs of growing radius. If the estimate has settled by 10 degrees
the domain is big enough and the failure is elsewhere; if it is still moving, the
model is being asked to steer a storm using a flow it cannot see.

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

    print(f"deep-layer steering over discs of growing radius\n")
    print(f"{'valid':<14}" + "".join(f"{r:>7.0f}deg" for r in radii)
          + f"{'change 8->max':>16}")
    print("-" * (14 + 10 * len(radii) + 16))

    drifts = []
    for t in times:
        try:
            f = pf.load_era5(era5, args.tc_id, t, args.n, None)
        except FileNotFoundError:
            print(f"{t:%Y-%m-%d %HZ}   (no file)")
            continue
        vals = [steering_at(f, r, pf.LEVELS) for r in radii]
        cells = "".join(f"{np.hypot(*v):>7.1f}   " if v else f"{'-':>10}"
                        for v in vals)
        # How much the answer is still moving between a comfortably interior
        # radius and the largest one the data supports.
        a = next((v for r, v in zip(radii, vals) if r >= 8.0 and v), None)
        b = next((v for r, v in reversed(list(zip(radii, vals))) if v), None)
        d = np.hypot(b[0] - a[0], b[1] - a[1]) if (a and b) else float('nan')
        drifts.append(d)
        print(f"{t:%Y-%m-%d %HZ} {cells}{d:>13.2f} m/s")

    if drifts:
        d = np.array([x for x in drifts if np.isfinite(x)])
        print(f"\nmedian change beyond 8 degrees: {np.median(d):.2f} m/s")
        print(f"for scale, a storm moves at 3-10 m/s, and the model follows its")
        print(f"own steering at 0.78x - a 22 % shortfall on 6 m/s is 1.3 m/s.")
        if np.median(d) > 0.5:
            print("\nThe estimate is still moving at the domain edge. Whatever")
            print("lies outside is steering information the model never sees,")
            print("and no amount of training recovers it.")
        else:
            print("\nThe estimate has settled well inside the domain, so the")
            print("boundary is not what these failures are about.")

    print("\nread this next to the disc-versus-square question: the polar grid")
    print("keeps r <= 10 while the Cartesian square reached 14.1 into its")
    print("corners, so any ridge to the north-east sat in the part that was")
    print("given up. Run tools/steering.py on the same case to see whether the")
    print("model's own steering matches ERA5's at the radius it can see.")


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
    p.add_argument("--n", type=int, default=161,
                   help="grid size to read; 161 keeps the corners the model "
                        "does not get to see")
    main(p.parse_args())
