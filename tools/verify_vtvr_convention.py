"""Determine which vt/vr sign convention the training dataset uses.

Why this exists
---------------
The dataset ships vt/vr precomputed; this project never produced them, so the
sign convention is not documented anywhere in the code. Inference has to invert
that rotation to hand u/v back to FCNV2, and guessing wrong yields a wind field
that is mirrored or rotated with no error raised anywhere - the forecast simply
comes out wrong in a way that is easy to mistake for a model problem.

There are more places to get a sign wrong than one would expect: whether the row
index runs north-to-south (ERA5 latitude is normally descending, which reverses
the sense of theta), whether vt is positive counter-clockwise, and whether vr is
positive outward. Rather than reason about all three, measure: a *_combined.nc
file holds u, v, vt and vr for the same points, so the convention can simply be
read off by trying all candidates and seeing which reproduces the stored values.

Usage (on the cluster, where the dataset lives)
-----------------------------------------------
    python tools/verify_vtvr_convention.py \
        /work/yungyun0721/TC_dataset/DLDA_data/ERA5_DLDA_data/labeled_and_obs_data_with_vt_vr/2007/<TCID>/<file>_combined.nc

Then copy the winning name into the `wind_convention` field of the model yaml
under onnx/.

The script deliberately reports the error of EVERY candidate, not just the best
one. If the winner is not orders of magnitude better than the rest, the geometry
assumed here does not match the data and the result should not be trusted.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DLAMPty_inference import WIND_CONVENTIONS  # noqa: E402

# Surface pair first: it is 2-D, so the geometry is easiest to reason about.
PAIRS = [('u10', 'v10', 'vt10', 'vr10'), ('u', 'v', 'vt', 'vr')]


def load(nc, name):
    import netCDF4
    if name not in nc.variables:
        return None
    return np.squeeze(np.asarray(nc.variables[name][:], dtype=np.float64))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    import netCDF4

    path = sys.argv[1]
    nc = netCDF4.Dataset(path)
    print(f"file: {path}")

    for u_n, v_n, vt_n, vr_n in PAIRS:
        u, v, vt, vr = (load(nc, n) for n in (u_n, v_n, vt_n, vr_n))
        if any(x is None for x in (u, v, vt, vr)):
            print(f"\n[{vt_n}/{vr_n}] missing one of {u_n},{v_n},{vt_n},{vr_n} - skipped")
            continue

        ny, nx = u.shape[-2:]
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx, indexing='ij')
        theta = np.arctan2(yy, xx)                      # same convention as latlon_to_polar
        sin_t, cos_t = np.sin(theta), np.cos(theta)

        # The centre point is degenerate (theta undefined) and the innermost
        # ring is nearly so; exclude it or it dominates the residual.
        r = np.hypot(xx, yy)
        keep = r > 2.0
        scale = float(np.nanmean(np.hypot(u, v)[..., keep])) or 1.0

        print(f"\n[{vt_n}/{vr_n}]  grid {ny}x{nx}, centre ({cy:g},{cx:g}), "
              f"{keep.sum()} points used, mean |wind| {scale:.2f} m/s")
        rows = []
        for name, (p, q, s, t) in WIND_CONVENTIONS.items():
            vt_try = p * u * sin_t + q * v * cos_t
            vr_try = s * u * cos_t + t * v * sin_t
            err = np.sqrt(np.nanmean((vt_try - vt)[..., keep] ** 2
                                     + (vr_try - vr)[..., keep] ** 2))
            rows.append((err, name))
        rows.sort()
        for err, name in rows:
            print(f"    {name:<20} rmse {err:10.4f} m/s   ({100*err/scale:6.1f}% of mean wind)")

        best_err, best = rows[0]
        runner_up = rows[1][0]
        print(f"    -> best: {best}")
        if best_err > 0.05 * scale:
            print("    WARNING: even the best candidate is a poor fit. The geometry "
                  "assumed here (centre, theta) probably does not match the data; "
                  "do NOT set wind_convention from this run.")
        elif runner_up < 5 * best_err:
            print("    WARNING: the runner-up is not clearly worse, so the answer is "
                  "ambiguous. Try another file, ideally one with a strong asymmetric "
                  "circulation.")
        else:
            print(f"    Set  wind_convention: {best}  in the model yaml.")

    nc.close()


if __name__ == '__main__':
    main()
