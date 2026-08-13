"""Does a northward wind survive the trip into polar coordinates and back?

The steering diagnostic on 202414W turned up something a training deficiency
cannot produce. Correlating the model's own deep-layer steering against the
motion of its own frame:

    ERA5    r(u) +0.93    r(v) +0.74
    polar   r(u) +0.99    r(v) -0.48

The zonal coupling is perfect - better than the analysis. The meridional one is
*inverted*. The model moved that storm north while its own winds said the
environment was pushing it south, which is not a model being imprecise; it is a
model being consistent about the wrong sign. And it is the same storm whose
forecast turned north when the real one kept going west.

A learned weakness degrades both components. Inverting one is what a sign error
looks like, and the candidate is the latitude axis: ERA5 stores it descending, so
row index increases southward, while `v` is positive northward. Anywhere theta is
built from row and column indices - the polar sampling, the vt/vr rotation - the
two conventions have to be reconciled, and reconciling them in one place but not
the other flips v and leaves u alone.

This tests exactly that, on the real transforms rather than a re-implementation:
build a field whose wind is uniform and purely northward, push it through
lat/lon -> polar -> vt/vr -> u/v -> lat/lon, and ask which way it points at the
end.

A clean result does not exonerate the model - it relocates the problem. If the
transforms preserve the sign then the inversion was learned, which means the
training data and the inference path disagree about the convention somewhere
upstream, and verify_vtvr_convention.py is the next thing to re-run.

Usage
-----
    python tools/meridional_check.py --yaml onnx/LLAT_polar_vtvr_v1.yaml
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def uniform_wind(n, u0, v0, res=0.25, lon0=130.0, lat0=20.0):
    """A domain with one constant wind everywhere, and its coordinate ramps.

    Latitude descends down the rows, as ERA5 stores it and as every array in
    this pipeline carries it. That convention is the thing under test, so it is
    written explicitly rather than inherited.
    """
    c = (n - 1) / 2.0
    lon = lon0 + (np.arange(n) - c) * res
    lat = lat0 - (np.arange(n) - c) * res
    lon2d, lat2d = np.meshgrid(lon, lat)
    u = np.full((n, n), float(u0))
    v = np.full((n, n), float(v0))
    return u, v, lon2d, lat2d


def main(args):
    from DLAMPty_inference import latlon_to_polar, polar_to_latlon, rotate_polar_wind

    R, Theta = args.R, args.Theta
    n = args.n
    r_max = (n - 1) / 2.0
    centre = (r_max, r_max)

    print(f"grid {n}x{n}, R={R}, Theta={Theta}, convention "
          f"{args.convention!r}\n")
    print(f"{'input (u,v)':>16} {'meaning':<22} {'recovered (u,v)':>18}  verdict")
    print("-" * 78)

    bad = []
    for (u0, v0), meaning in (((0.0, 10.0), "pure northward"),
                              ((0.0, -10.0), "pure southward"),
                              ((-10.0, 0.0), "pure westward"),
                              ((10.0, 0.0), "pure eastward")):
        u, v, lon2d, lat2d = uniform_wind(n, u0, v0)
        stack = np.stack([u, v], axis=-1)

        polar, _r, theta_deg = latlon_to_polar(stack, R=R, Theta=Theta,
                                               r_max=r_max, center_xy=centre)
        theta = np.deg2rad(theta_deg)          # (Theta,), as the rotation wants
        rot = rotate_polar_wind(polar.copy(), theta, 0, 1, args.convention,
                                inverse=False)
        back_polar = rotate_polar_wind(rot, theta, 0, 1, args.convention,
                                       inverse=True)
        out = polar_to_latlon(back_polar, output_shape=(n, n), r_max=r_max,
                              center_xy=centre, fill_value=np.nan)
        ru = float(np.nanmean(out[..., 0]))
        rv = float(np.nanmean(out[..., 1]))

        ok = (abs(ru - u0) < 0.5) and (abs(rv - v0) < 0.5)
        flipped = (abs(ru - u0) < 0.5) and (abs(rv + v0) < 0.5) and v0 != 0
        verdict = "ok" if ok else ("V IS INVERTED" if flipped else "WRONG")
        if not ok:
            bad.append((meaning, verdict))
        print(f"{f'({u0:+.0f}, {v0:+.0f})':>16} {meaning:<22} "
              f"{f'({ru:+.2f}, {rv:+.2f})':>18}  {verdict}")

    print()
    if not bad:
        print("The transforms preserve every direction, so the inverted")
        print("meridional correlation was not produced here. That relocates the")
        print("question rather than answering it: the model learned the")
        print("inversion, which means the training data and this path disagree")
        print("about the convention somewhere upstream. Re-run")
        print("tools/verify_vtvr_convention.py against the training files, not")
        print("just one forecast file.")
    else:
        print("A direction did not survive the round trip:")
        for meaning, verdict in bad:
            print(f"  {meaning}: {verdict}")
        print("\nThat is a coded sign error, not a learned one, and it would")
        print("explain a model that moves north while its winds push south.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=81)
    p.add_argument("--R", type=int, default=41)
    p.add_argument("--Theta", type=int, default=180)
    p.add_argument("--convention", default="ccw_inward_flip",
                   help="the one measured by verify_vtvr_convention.py and "
                        "recorded in the model card")
    main(p.parse_args())
