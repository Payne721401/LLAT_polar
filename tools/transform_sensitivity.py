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

# Full names and units, because "tp" and "4.9% lost" on a y-axis tell a reader
# neither which field it is nor what was measured.
NICE = {
    'u':          ('zonal wind u', 'm s$^{-1}$'),
    'v':          ('meridional wind v', 'm s$^{-1}$'),
    'msl':        ('mean sea level pressure', 'Pa'),
    'tp':         ('total precipitation', 'm'),
    'tcwv':       ('total column water vapour', 'kg m$^{-2}$'),
    't2m':        ('2 m temperature', 'K'),
    'vorticity':  ('relative vorticity  dv/dx - du/dy', '10$^{-5}$ s$^{-1}$'),
    'divergence': ('divergence  du/dx + dv/dy', '10$^{-5}$ s$^{-1}$'),
}


def nice(name):
    return NICE.get(name, (name, ''))


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

    # ── Signed bias by ring, which is the question this file was reopened for ──
    #
    # The RMS above says how much the round trip disturbs a field. It cannot say
    # whether the disturbance has a direction, and that is what matters here:
    # the polar forecasts carry an msl bias of -125 Pa inside 100 km at +24 h,
    # six times the Cartesian model's, and the transform runs twice per step -
    # 128 times out to 192 h, each on the previous output. If the round trip
    # alone has a core bias, part of that number is the pipeline rather than the
    # model, and the fix is a different one.
    #
    # Rings in kilometres, with the cos(lat) factor, matching
    # tools/season_radial_rmse.py so the two tables can be read against each
    # other. The innermost ring is where a polar mesh is most degenerate:
    # Theta points all converge on the pole, so the interpolation there is
    # doing the most work and has the most opportunity to be one-sided.
    edges = [float(x) for x in args.rings.split(',')]
    names = [f"{edges[k]:.0f}-{edges[k+1]:.0f}" for k in range(len(edges) - 1)]
    dxkm = (lon2d - lon2d[n // 2, n // 2]) * 111.32 * np.cos(np.deg2rad(lat2d))
    dykm = (lat2d - lat2d[n // 2, n // 2]) * 111.32
    rkm = np.hypot(dxkm, dykm)

    print()
    print("signed bias of the round trip alone - no model, no forecast")
    print(f"{'field':<14}" + "".join(f"{nm + ' km':>16}" for nm in names))
    print("-" * (14 + 16 * len(names)))
    for name, (a, b) in fields.items():
        line = f"{name:<14}"
        for k in range(len(names)):
            sel = (inside & np.isfinite(a) & np.isfinite(b)
                   & (rkm >= edges[k]) & (rkm < edges[k + 1]))
            line += (f"{float(np.mean(b[sel] - a[sel])):>16.5g}" if sel.any()
                     else f"{'-':>16}")
        print(line)
    print()
    print("  One pass. The forecast applies this 128 times out to 192 h, each")
    print("  on the output of the last, so a bias here does not stay this size.")
    print("  Compare against the model's own core bias from season_radial_rmse:")
    print("  msl -125.6 Pa inside 100 km at +24 h, against the Cartesian -20.5.")

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

    if args.out is None:
        args.out = os.path.join(
            "analysis", "figures", "transform",
            f"sensitivity_R{args.R}_T{args.Theta}_L{args.level}"
            + ("_rot" if args.rotate else "")
            + ("_square" if args.square else "") + ".png")
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
        if not args.square:
            a = np.where(inside, a, np.nan)
        # b keeps its NaN corners either way. With --square that is the point:
        # 23.4 % of the frame has no polar data at all, and masking the original
        # to match hides the single largest thing the grid costs.
        d = b - a
        # A field centred on zero (vorticity, divergence) wants a symmetric
        # diverging scale; one with a large offset (msl near 100,500 Pa) is
        # rendered a single flat colour by it. Deciding per field rather than
        # globally, which drew the msl row as a solid block.
        lo, hi = (float(np.nanpercentile(a, 1)), float(np.nanpercentile(a, 99)))
        if lo < 0 < hi and abs(lo) / max(hi, 1e-9) > 0.2:
            vm = max(abs(lo), abs(hi))
            lo, hi, cmap = -vm, vm, 'RdBu_r'
        else:
            cmap = 'viridis'
        # The difference is shown as a percentage of the field's own RMS, on a
        # scale fixed at +/-25 % for every row. A raw difference needs its own
        # colourbar per row, in that row's units, and comparing rows then means
        # dividing two numbers in your head; this way a strong colour means the
        # same thing everywhere on the figure.
        fa = float(np.sqrt(np.nanmean(np.where(np.isfinite(a), a, 0) ** 2)))
        dpct = 100.0 * d / fa if fa else d
        for col, (arr, ttl, kwargs) in enumerate([
                (a, "original", dict(cmap=cmap, vmin=lo, vmax=hi)),
                (b, "after polar round trip", dict(cmap=cmap, vmin=lo, vmax=hi)),
                (dpct, "error, % of field RMS",
                 dict(cmap='PuOr_r', vmin=-25, vmax=25))]):
            im = ax[row][col].imshow(arr, origin='lower', cmap=kwargs['cmap'],
                                     vmin=kwargs['vmin'], vmax=kwargs['vmax'])
            if col == 2:
                # Outline the original underneath. Interpolation error lives on
                # sharp gradients, and seeing the two together is the difference
                # between "speckle" and "it traces the rainbands".
                with np.errstate(invalid='ignore'):
                    ax[row][col].contour(np.nan_to_num(a), levels=6,
                                         colors='0.35', linewidths=0.4,
                                         alpha=0.7)
            if row == 0:
                ax[row][col].set_title(ttl)
            fig.colorbar(im, ax=ax[row][col], fraction=0.046,
                         label='%' if col == 2 else '')
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
    p.add_argument("--rings", default="0,100,300,600,1110",
                   help="ring boundaries in km for the signed-bias table, "
                        "matching season_radial_rmse so the two can be read "
                        "against each other")
    p.add_argument("--rotate", action="store_true",
                   help="include the u/v -> vt/vr -> u/v rotation the vt_vr "
                        "model cards use")
    p.add_argument("--convention", default="ccw_inward_flip")
    p.add_argument("--square", action="store_true",
                   help="draw the whole 81x81 frame instead of just the disc. "
                        "The round-trip and difference columns then show the "
                        "corners as blank, which is 23.4 %% of the domain the "
                        "polar grid does not represent at all - the largest "
                        "single thing it costs, and invisible when both columns "
                        "are masked to the disc")
    p.add_argument("--out", default=None,
                   help="defaults to a name carrying R, Theta, the level and "
                        "whether --rotate was used, so the t180 and t360 runs "
                        "do not overwrite each other")
    main(p.parse_args())
