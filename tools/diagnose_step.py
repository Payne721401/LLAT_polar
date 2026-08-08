"""Instrument one forecast step and report where NaN enters.

Written for a failure that surfaced only as downstream symptoms: "Mean of empty
slice" from lonlat_uniformizer, then numpy.nanmax raising on an empty array deep
inside xarray_regrid. Neither says which stage produced the NaN, and the chain
has six of them, so reasoning about it is guesswork. This measures each boundary
instead.

Reports, for every stage, the fraction of NaN in the coordinate channels and in
the weather channels separately - they fail for different reasons. Some NaN is
expected: polar_to_latlon leaves the corners undefined, 23.4 % of the frame. What
is not expected is a coordinate channel going entirely NaN, because the TC centre
is recovered by averaging it.

Usage
-----
    python tools/diagnose_step.py \
        --model-yaml onnx/LLAT_polar_vtvr_v1.yaml \
        --ic /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty/202421W_2024102500_combined.nc
"""
import argparse
import os
import warnings

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

from DLAMPty_inference import (DLAMPty_model, latlon_to_polar,  # noqa: E402
                               polar_to_latlon)
from utils.data_processor import lonlat_uniformizer  # noqa: E402


def report(tag, arr, coord_idx=None):
    a = np.asarray(arr, dtype=float)
    n = a.size
    nan = int(np.isnan(a).sum())
    line = f"  {tag:<34} shape {str(a.shape):<22} NaN {100*nan/n:>5.1f}%"
    if coord_idx is not None:
        for name, i in coord_idx:
            c = a[..., i]
            frac = 100 * np.isnan(c).sum() / c.size
            finite = c[np.isfinite(c)]
            rng = (f"{finite.min():.2f}..{finite.max():.2f}" if finite.size
                   else "ALL NaN")
            line += f"\n      {name:<6} NaN {frac:>5.1f}%   range {rng}"
    print(line)


def main(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = DLAMPty_model(args.model_yaml, root_dir=root, device='cpu')
    m.initialize()
    n = m.cartesian_n
    coords = [('lon', -2), ('lat', -1)]

    print(f"\nIC: {args.ic}")
    with xr.open_dataset(args.ic) as ds:
        ny, nx = ds.sizes['latitude'], ds.sizes['longitude']
        if (ny, nx) != (n, n):
            oy, ox = (ny - n) // 2, (nx - n) // 2
            ds = ds.isel(latitude=np.arange(oy, oy + n),
                         longitude=np.arange(ox, ox + n))
            print(f"  cropped {ny}x{nx} -> {n}x{n}")
        up, sfc = m.IC_from_xarray_to_npy(ds)

    print("\n--- stage by stage ---")
    report("1 IC surface", sfc, coords)
    report("1 IC upper", up)

    kw = dict(R=m.polar_R, Theta=m.polar_theta, r_max=m.r_max_px,
              center_xy=m.center_xy)
    pu, _, theta_deg = latlon_to_polar(up, **kw)
    ps, _, _ = latlon_to_polar(sfc, **kw)
    report("2 after latlon_to_polar (sfc)", ps, coords)
    report("2 after latlon_to_polar (upper)", pu)

    if m.wind_representation == 'vt_vr':
        m._rotate_wind(pu, ps, np.deg2rad(theta_deg), inverse=False)
        report("3 after u/v -> vt/vr", ps, coords)

    nu, ns = m.normalize(pu, ps)
    report("4 after normalise", ns, coords)

    out = m.model.run(None, {"input_upper": nu[None].astype(np.float32),
                             "input_surface": ns[None].astype(np.float32)})
    ou, os_ = out[0].squeeze(), out[1].squeeze()
    report("5 model output (polar, normalised)", os_, coords)

    ou, os_ = m.normalize(ou, os_, reverse=True)
    report("6 after denormalise", os_, coords)

    if m.wind_representation == 'vt_vr':
        m._rotate_wind(ou, os_, np.deg2rad(theta_deg), inverse=True)
        report("7 after vt/vr -> u/v", os_, coords)

    back = dict(output_shape=(n, n), r_max=m.r_max_px, center_xy=m.center_xy,
                mode="bilinear", fill_value=np.nan)
    cs = polar_to_latlon(os_, **back)
    report("8 after polar_to_latlon", cs, coords)

    print("\n--- centre recovery ---")
    raw_lon, raw_lat = cs[:, :, -2], cs[:, :, -1]
    for name, a in (('lon', raw_lon), ('lat', raw_lat)):
        col = np.isnan(a).all(axis=0 if name == 'lon' else 1).sum()
        print(f"  {name}: {100*np.isnan(a).mean():.1f}% NaN, "
              f"{col} fully-NaN {'columns' if name == 'lon' else 'rows'}")
    lon, lat = lonlat_uniformizer(raw_lon, raw_lat, True, m.specify_resolution)
    print(f"  uniformizer -> lon centre {np.nanmean(lon):.4f}, "
          f"lat centre {np.nanmean(lat):.4f}")
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        print("  ^^ NaN in the reconstructed axis: this is the failure. Every "
              "downstream error follows from it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-yaml", default="onnx/LLAT_polar_vtvr_v1.yaml")
    p.add_argument("--ic", required=True, help="a *_combined.nc initial condition")
    main(p.parse_args())
