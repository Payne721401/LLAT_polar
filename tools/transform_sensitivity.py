"""What does the polar round trip cost each field, and which fields notice most?

rmin_test --map answered this for one scalar and found only 4 % of the error
axisymmetric, which ruled the transform out as the source of concentric rings.
It does not answer the sharper question: a derivative amplifies interpolation
error by roughly 1/dx, so vorticity computed from round-tripped wind can be far
worse than the wind itself, and precipitation is nearly all small-scale
structure with nothing smooth for bilinear sampling to hold on to. Those are the
fields to look at, and a scalar RMS on msl says nothing about them.

The order of operations is the point. Vorticity is computed AFTER the round trip,
from round-tripped u and v, because that is what the pipeline does - the model
predicts wind on the polar grid and everything derived comes later. Round-tripping
a vorticity field that was computed first would measure a different thing and
flatter the transform.

Uses the real latlon_to_polar / polar_to_latlon / rotate_polar_wind from
DLAMPty_inference rather than a re-implementation, so what is measured is what
runs. --rotate adds the vt/vr conversion the vt_vr model cards use, which is a
second interpolation-sensitive step: the rotation is exact, but it makes the two
wind components functions of theta, and the round trip samples them separately.

No model, no weights, no GPU: this is a property of the grid.

Usage
-----
    E=/wk2/yungyun/FCNV2_TC
    python tools/transform_sensitivity.py \\
        --nc $E/202421W/ERA5/for_DLAMPty/202421W_2024102700_combined.nc \\
        --R 40 --Theta 180 --rotate \\
        --out analysis/figures/transform/sensitivity_t180.png

    # the same file on the t360 grid, to price the azimuthal density
    python tools/transform_sensitivity.py --nc ... --R 40 --Theta 360 --rotate \\
        --out analysis/figures/transform/sensitivity_t360.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

M_PER_DEG = 111_320.0


def curl_div(u, v, lon, lat):
    """Relative vorticity and divergence on the lat/lon grid, in s^-1.

    The same formulation plot_forecast uses, so a difference seen here is the
    difference seen there. Spacing comes from the coordinate arrays rather than
    being assumed, and the cos(lat) factor on longitude matters at 20 N.
    """
    dy = np.gradient(lat, axis=0) * M_PER_DEG
    dx = np.gradient(lon, axis=1) * M_PER_DEG * np.cos(np.deg2rad(lat))
    vort = np.gradient(v, axis=1) / dx - np.gradient(u, axis=0) / dy
    div = np.gradient(u, axis=1) / dx + np.gradient(v, axis=0) / dy
    return vort, div


def main(args):
    import xarray as xr
    from DLAMPty_inference import (latlon_to_polar, polar_to_latlon,
                                   rotate_polar_wind)

    with xr.open_dataset(os.path.expanduser(args.nc)) as ds:
        def get(name, level=None):
            a = ds[name]
            if level is not None and 'level' in a.dims:
                a = a.sel(level=level)
            return np.squeeze(a.values).astype(float)

        n0 = ds.sizes['latitude']
        n = args.crop or n0
        o = (n0 - n) // 2
        sl = (slice(o, o + n), slice(o, o + n))

        u = get('u', args.level)[sl]
        v = get('v', args.level)[sl]
        lon2d, lat2d = np.meshgrid(ds.longitude.values[o:o + n],
                                   ds.latitude.values[o:o + n])
        scalars = {}
        for name in args.vars.split(','):
            name = name.strip()
            if not name or name in ('u', 'v'):
                continue
            if name not in ds:
                print(f"  note: {name} not in the file; skipped")
                continue
            a = get(name, args.level if 'level' in ds[name].dims else None)
            scalars[name] = a[sl]

    r_max = (n - 1) / 2.0
    centre = (r_max, r_max)
    kw = dict(R=args.R, Theta=args.Theta, r_max=r_max, center_xy=centre)
    back_kw = dict(output_shape=(n, n), r_max=r_max, center_xy=centre,
                   mode="bilinear", fill_value=np.nan)

    # Wind first, together, because vt/vr couples the two components.
    stack = np.stack([u, v], axis=-1)
    polar, _r, theta_deg = latlon_to_polar(stack, **kw)
    if args.rotate:
        th = np.deg2rad(theta_deg)
        polar = rotate_polar_wind(polar, th, 0, 1, args.convention, inverse=False)
        polar = rotate_polar_wind(polar, th, 0, 1, args.convention, inverse=True)
    rt = polar_to_latlon(polar, **back_kw)
    u2, v2 = rt[..., 0], rt[..., 1]

    fields = {'u': (u, u2), 'v': (v, v2)}
    for name, a in scalars.items():
        p, _r, _t = latlon_to_polar(a[..., None], **kw)
        fields[name] = (a, polar_to_latlon(p, **back_kw)[..., 0])

    # Derived AFTER the round trip, which is the order the pipeline uses.
    vort, div = curl_div(u, v, lon2d, lat2d)
    vort2, div2 = curl_div(u2, v2, lon2d, lat2d)
    fields['vorticity'] = (vort * 1e5, vort2 * 1e5)
    fields['divergence'] = (div * 1e5, div2 * 1e5)

    # Only inside the disc: outside it the model has no output at all, so an
    # error there is not the transform's doing.
    yy, xx = np.meshgrid(np.arange(n) - r_max, np.arange(n) - r_max,
                         indexing='ij')
    inside = np.hypot(xx, yy) <= r_max

    print(f"{os.path.basename(args.nc)}   {n}x{n} -> R={args.R}, "
          f"Theta={args.Theta}"
          f"{', with vt/vr rotation' if args.rotate else ''}\n")
    print(f"{'field':<14}{'rms field':>12}{'rms error':>12}{'error/field':>13}")
    print("-" * 51)
    rows = {}
    for name, (a, b) in fields.items():
        m = inside & np.isfinite(a) & np.isfinite(b)
        fa = float(np.sqrt(np.nanmean(a[m] ** 2)))
        fe = float(np.sqrt(np.nanmean((b[m] - a[m]) ** 2)))
        rows[name] = fe / fa if fa else np.nan
        print(f"{name:<14}{fa:>12.4g}{fe:>12.4g}{100 * rows[name]:>12.1f}%")

    wind = max(rows.get('u', 0), rows.get('v', 0))
    print()
    if wind and rows.get('vorticity'):
        amp = rows['vorticity'] / wind
        print(f"Vorticity loses {amp:.1f}x as much of itself as the wind it is")
        print("computed from. A first derivative divides by the grid spacing, so")
        print("interpolation error that is invisible in u and v is not invisible")
        print("in anything differentiated from them.", end=" ")
        if amp > 3:
            print("At this ratio, any")
            print("vorticity figure drawn from a polar forecast is showing a")
            print("substantial amount of resampling rather than weather.")
        else:
            print("The amplification here is")
            print("modest enough that vorticity plots remain readable.")
    if 'tp' in rows:
        print(f"\nPrecipitation loses {100 * rows['tp']:.0f}% of itself. It is "
              f"almost all\nsmall-scale structure, with nothing smooth for "
              f"bilinear sampling to\nhold on to - the opposite of msl.")

    if not args.out:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = [k for k in args.order.split(',') if k.strip() in fields]
    order = order or list(fields)
    fig, ax = plt.subplots(len(order), 3, figsize=(13, 3.6 * len(order)),
                           squeeze=False)
    for row, name in enumerate(order):
        a, b = fields[name]
        a = np.where(inside, a, np.nan)
        b = np.where(inside, b, np.nan)
        d = b - a
        vm = float(np.nanpercentile(np.abs(a), 99))
        dm = float(np.nanpercentile(np.abs(d[np.isfinite(d)]), 99)) or 1.0
        for col, (arr, ttl, kwargs) in enumerate([
                (a, "original", dict(cmap='RdBu_r', vmin=-vm, vmax=vm)),
                (b, "round trip", dict(cmap='RdBu_r', vmin=-vm, vmax=vm)),
                (d, "difference", dict(cmap='PuOr_r', vmin=-dm, vmax=dm))]):
            im = ax[row][col].imshow(arr, origin='lower', cmap=kwargs['cmap'],
                                     vmin=kwargs['vmin'], vmax=kwargs['vmax'])
            if row == 0:
                ax[row][col].set_title(ttl)
            fig.colorbar(im, ax=ax[row][col], fraction=0.046)
        ax[row][0].set_ylabel(f"{name}\n{100 * rows[name]:.1f}% lost",
                              fontsize=9)
    fig.suptitle(f"polar round trip, R={args.R} Theta={args.Theta}"
                 f"{' + vt/vr' if args.rotate else ''} — "
                 f"derived quantities computed AFTER the trip")
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nc", required=True, help="a *_combined.nc")
    p.add_argument("--vars", default="msl,tp,tcwv,t2m",
                   help="scalar channels to round-trip; u and v are always "
                        "done, together, because vt/vr couples them")
    p.add_argument("--order", default="vorticity,divergence,tp,u,msl",
                   help="which fields to draw, in order. The default leads "
                        "with the differentiated ones, which is where the "
                        "question is")
    p.add_argument("--level", type=int, default=850,
                   help="pressure level for the upper-air fields")
    p.add_argument("--crop", type=int, default=81,
                   help="centre-crop first; the saved file is 161x161 and the "
                        "model runs on 81x81")
    p.add_argument("--R", type=int, default=40)
    p.add_argument("--Theta", type=int, default=180)
    p.add_argument("--rotate", action="store_true",
                   help="include the u/v -> vt/vr -> u/v rotation the vt_vr "
                        "model cards use")
    p.add_argument("--convention", default="ccw_inward_flip")
    p.add_argument("--out", default="analysis/figures/transform/sensitivity.png")
    main(p.parse_args())
