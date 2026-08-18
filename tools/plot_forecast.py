"""Compare forecasts side by side: one column per experiment, one row per field.

A rewrite of the lab's kong_Rey_animation_4plots notebook, which grew into a
personal workspace: nineteen cells, around thirty absolute paths, and branches
for DA cycling, nudging and conference figures interleaved with the comparison
itself. The panels it draws are worth keeping; the surrounding machinery is not.

Differences that matter
-----------------------
Self-contained. Coastlines come from the `landmask` channel that every forecast
already carries, so there is no coastline file to locate and the outline is by
construction aligned with the data. No cartopy either - the domain is 20x20
degrees, where a plate-carree axis is just a lat/lon axis.

pcolormesh, never contourf. contourf interpolates between real values and fill
values, inventing smooth rainbow structure across the boundary of the polar disc
that does not exist in the data.

NaN is drawn as blank. Outside the polar disc the model has no output; plotting
it as 0 is what made earlier figures look like they had a bad outer ring.

Usage
-----
    python tools/plot_forecast.py \
        --run "LLAT polar=~/LLAT_polar_runs/202421W/2_way_circle_couple_model_LLAT_polar_vtvr_v1/start_from_2024102500" \
        --era5 /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty \
        --tc-id 202421W --init 2024102500 --lead 24 \
        --out fig_024h.png

--run may be repeated to add columns. --era5 adds a truth column first.
"""
import argparse
import datetime
import os

import numpy as np

# Fallback channel order, used only when a run predates run_meta.yaml.
# run_coupled_forecast writes that file precisely so this does not have to be
# trusted: indexing by position means a changed model card would otherwise shift
# every field by one, and nothing would look wrong.
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SFC = ['u10', 'v10', 't2m', 'd2m', 'msl', 'sp', 'tcwv', 'tp', 'mtnlwrf',
       'sst_filled', 'f', 'solar', 'hgt', 'landmask', 'diurnal_sin',
       'diurnal_cos', 'doy_sin', 'doy_cos', 'lon', 'lat']
UPPER = ['u', 'v', 't', 'q', 'z', 'w']


def read_meta(run_dir):
    """Channel layout for a run, from its run_meta.yaml when there is one."""
    import yaml

    p = os.path.join(os.path.expanduser(run_dir), 'run_meta.yaml')
    if not os.path.exists(p):
        print(f"  note: {os.path.basename(run_dir)} has no run_meta.yaml, "
              "assuming the default channel order")
        return dict(surface_vars=SFC, upper_vars=UPPER, pressure_levels=LEVELS)
    with open(p, encoding='utf-8') as f:
        m = yaml.safe_load(f)
    return dict(surface_vars=m['surface_vars'], upper_vars=m['upper_vars'],
                pressure_levels=m['pressure_levels'])


class Field:
    """One experiment at one valid time, addressed by variable name."""

    def __init__(self, upper, sfc, meta=None):
        meta = meta or dict(surface_vars=SFC, upper_vars=UPPER, pressure_levels=LEVELS)
        self.upper, self.sfc = upper, sfc
        self.sfc_names = list(meta['surface_vars'])
        self.upper_names = list(meta['upper_vars'])
        self.levels = list(meta['pressure_levels'])
        if len(self.sfc_names) != sfc.shape[-1]:
            raise ValueError(
                f"run_meta lists {len(self.sfc_names)} surface channels but the "
                f"array has {sfc.shape[-1]}")
        self.lon = sfc[..., self.sfc_names.index('lon')]
        self.lat = sfc[..., self.sfc_names.index('lat')]

    def s(self, name):
        return self.sfc[..., self.sfc_names.index(name)]

    def u(self, name, level):
        return self.upper[self.levels.index(level), :, :,
                          self.upper_names.index(name)]

    def vorticity(self, level):
        """Relative vorticity dv/dx - du/dy on the lat/lon grid.

        Spacing is taken from the coordinate channels rather than assumed, so
        this stays correct if the domain resolution ever changes. The cos(lat)
        factor matters at these latitudes: one degree of longitude is about 4%
        shorter at 20N than at the equator.
        """
        u, v = self.u('u', level), self.u('v', level)
        m_per_deg = 111_320.0
        dy = np.gradient(self.lat, axis=0) * m_per_deg
        dx = np.gradient(self.lon, axis=1) * m_per_deg * np.cos(np.deg2rad(self.lat))
        return np.gradient(v, axis=1) / dx - np.gradient(u, axis=0) / dy


def forecast_dir(run_dir):
    return os.path.join(os.path.expanduser(run_dir), 'DLAMPty', 'forecast')


def available_leads(run_dir):
    """Forecast hours actually present, so --lead need not be guessed."""
    import glob
    import re

    out = []
    for f in glob.glob(os.path.join(forecast_dir(run_dir), 'output_sfc_*h.npy')):
        m = re.search(r'output_sfc_(\d+)h\.npy$', f)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_run(run_dir, lead, meta=None):
    d = forecast_dir(run_dir)
    up = np.load(os.path.join(d, f"output_upper_{lead:0>3}h.npy"))
    sfc = np.load(os.path.join(d, f"output_sfc_{lead:0>3}h.npy"))
    return Field(up, sfc, meta)


_NOTED = set()


def load_era5(era5_dir, tc_id, valid_time, n, meta=None):
    """Truth from the combined.nc at the valid time, cropped to the model domain."""
    import xarray as xr

    p = os.path.join(os.path.expanduser(era5_dir),
                     f"{tc_id}_{valid_time:%Y%m%d%H}_combined.nc")
    with xr.open_dataset(p) as ds:
        ny, nx = ds.sizes['latitude'], ds.sizes['longitude']
        if (ny, nx) != (n, n):
            oy, ox = (ny - n) // 2, (nx - n) // 2
            ds = ds.isel(latitude=np.arange(oy, oy + n),
                         longitude=np.arange(ox, ox + n))
        names = (meta or {}).get('surface_vars', SFC)
        upper_names = (meta or {}).get('upper_vars', UPPER)
        up = np.stack([np.squeeze(ds[v].values) for v in upper_names], axis=-1)

        # A combined.nc holds raw ERA5. The derived channels - landmask, f,
        # solar, sst_filled, the time encodings - are produced by
        # calc_additional_vars during the forecast and are simply absent here.
        # Rather than run that (it regrids a global land mask and evaluates
        # solar position per point), take what exists and leave the rest NaN;
        # of the derived fields only landmask is used for plotting, and that
        # can be recovered exactly, see below.
        shape = np.squeeze(ds[upper_names[0]].values).shape[-2:]
        missing = []
        cols = []
        for v in names[:-2]:
            if v in ds:
                cols.append(np.squeeze(ds[v].values))
            else:
                missing.append(v)
                cols.append(np.full(shape, np.nan))
        sfc = np.stack(cols, axis=-1)

        # ERA5 sst is undefined over land, so its missing-value mask IS the land
        # mask - at the source resolution and exactly aligned, with no regridding.
        if 'landmask' in missing and 'sst' in ds:
            sfc[..., names.index('landmask')] = np.isnan(
                np.squeeze(ds['sst'].values)).astype(float)
            missing.remove('landmask')
        # Once per distinct set, not once per file. track_error and intensity
        # each open one ERA5 file per lead, and thirty identical notices bury the
        # table they are supposed to preface.
        if missing and tuple(missing) not in _NOTED:
            _NOTED.add(tuple(missing))
            print(f"  note: ERA5 file has no {', '.join(missing)}; "
                  "left blank (derived during the forecast, not stored)")

        lon, lat = np.meshgrid(ds.longitude.values, ds.latitude.values)
        sfc = np.concatenate([sfc, lon[..., None], lat[..., None]], axis=-1)
    return Field(up, sfc, meta)


# Each panel: (row label, what to shade, colormap, how to draw the wind).
# Kept as data so adding or reordering rows needs no code change.
PANELS = [
    dict(label="10 m wind + MSLP",
         shade=lambda f: np.hypot(f.s('u10'), f.s('v10')),
         cmap="YlGnBu", unit="m s$^{-1}$",
         zero_based=True,
         wind=('stream', lambda f: (f.s('u10'), f.s('v10'))),
         contour=lambda f: f.s('msl') / 100.0),
    dict(label="Precipitation",
         shade=lambda f: f.s('tp') * 1000.0,
         cmap="GnBu", unit="mm", zero_based=True,
         wind=('quiver', lambda f: (f.s('u10'), f.s('v10')))),
    dict(label="850 hPa vorticity",
         shade=lambda f: f.vorticity(850) * 1e5,
         cmap="RdBu_r", unit="10$^{-5}$ s$^{-1}$", sym=True,
         wind=('quiver', lambda f: (f.u('u', 850), f.u('v', 850)))),
    dict(label="700 hPa omega",
         shade=lambda f: f.u('w', 700),
         cmap="BrBG", unit="Pa s$^{-1}$", sym=True,
         wind=('stream', lambda f: (f.u('u', 700), f.u('v', 700)))),
    dict(label="500 hPa wind + TCWV",
         shade=lambda f: f.s('tcwv'),
         cmap="YlGnBu", unit="kg m$^{-2}$",
         wind=('stream', lambda f: (f.u('u', 500), f.u('v', 500)))),
    # ── Beyond the default five, selected with --panels ──────────────────
    #
    # Each exists for one open question, and none needs a forecast rerun: they
    # are all derived from channels already saved.
    #
    # The subtropical ridge is defined by z500 contours, and where its western
    # edge sits decides whether a storm recurves. The TCWV panel above shows
    # moisture, which does not mark the ridge at all, so the one field bearing
    # directly on the track bias was not being drawn.
    dict(label="500 hPa height",
         shade=lambda f: f.u('z', 500) / 9.80665,
         cmap="Spectral_r", unit="m",
         wind=('stream', lambda f: (f.u('u', 500), f.u('v', 500))),
         contour=lambda f: f.u('z', 500) / 9.80665),
    # Deep-layer shear is the classic constraint on intensification, and the
    # measured error is that intensification arrives about 60 h late. A
    # systematically stronger shear would be the mechanism.
    dict(label="200-850 shear",
         shade=lambda f: np.hypot(f.u('u', 200) - f.u('u', 850),
                                  f.u('v', 200) - f.u('v', 850)),
         cmap="magma_r", unit="m s$^{-1}$", zero_based=True,
         wind=('quiver', lambda f: (f.u('u', 200) - f.u('u', 850),
                                    f.u('v', 200) - f.u('v', 850)))),
    # Warm core, as an anomaly against the panel's own areal mean so the colours
    # mean the same thing at every lead and latitude. This is the field that
    # separates a tropical cyclone from an extratropical one - a distinction
    # once asserted here from centre-point scalars, which cannot settle it.
    dict(label="300 hPa T anomaly",
         shade=lambda f: f.u('t', 300) - np.nanmean(f.u('t', 300)),
         cmap="RdBu_r", unit="K", sym=True,
         wind=('quiver', lambda f: (f.u('u', 300), f.u('v', 300)))),
]


# Rows drawn when --panels is not given. The rest are opt-in so that every
# existing command, figure and animation keeps the layout it had.
N_DEFAULT = 5


def mask_outside(field, arr, radius_deg):
    """Blank everything beyond radius_deg of the domain centre."""
    if not radius_deg:
        return arr
    n = arr.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    out = np.array(arr, dtype=float)
    out[np.hypot(xx, yy) * res > radius_deg] = np.nan
    return out


def draw(fig, ax, field, panel, radius, vlim, first_col, quiver_scale):
    lon, lat = field.lon, field.lat
    z = mask_outside(field, panel['shade'](field), radius)

    kw = dict(cmap=panel['cmap'], shading='auto')
    if panel.get('sym'):
        kw.update(vmin=vlim[0], vmax=vlim[1])
    else:
        # Anchor at zero only where zero is meaningful (wind speed, rainfall).
        # Forcing it on a field with a large baseline - TCWV sits around
        # 50 kg m^-2 - spends the whole colour range on values that never occur
        # and renders every panel the same shade.
        kw.update(vmin=0 if panel.get('zero_based') else vlim[0], vmax=vlim[1])
    mesh = ax.pcolormesh(lon, lat, z, **kw)

    kind, get = panel['wind']
    u, v = (mask_outside(field, a, radius) for a in get(field))
    if kind == 'stream':
        # streamplot requires strictly increasing coordinates, but latitude
        # descends with row index here (the ERA5 convention this project
        # inherits), so the rows are flipped for the call. pcolormesh above has
        # no such constraint, which is why only this branch needs it.
        y = lat[:, 0]
        us, vs = np.nan_to_num(u), np.nan_to_num(v)
        if y[0] > y[-1]:
            y, us, vs = y[::-1], us[::-1], vs[::-1]
        # streamplot also cannot integrate across NaN; the zeros above are only
        # for the integration, the shading still shows those cells as blank.
        ax.streamplot(lon[0, :], y, us, vs,
                      color='0.25', linewidth=0.6, density=0.9, arrowsize=0.7)
    else:
        k = (slice(None, None, 4), slice(None, None, 4))
        # Scale from the row's own wind speed so arrows stay legible whatever
        # the level and variable; a fixed scale suits one panel and ruins others.
        ax.quiver(lon[k], lat[k], u[k], v[k], scale=quiver_scale, width=0.004,
                  color='0.25')

    if panel.get('contour') is not None:
        c = mask_outside(field, panel['contour'](field), radius)
        cs = ax.contour(lon, lat, c, levels=8, colors='k', linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%d')

    # Coastline straight from the forecast's own land mask: no external file,
    # and it can never be misaligned with the data. Skipped rather than faked
    # when a column has no mask at all.
    land = field.s('landmask')
    if not np.all(np.isnan(land)):
        ax.contour(lon, lat, np.nan_to_num(land), levels=[0.5],
                   colors='darkslategray', linewidths=1.0)

    ax.set_aspect('equal')
    if first_col:
        ax.set_ylabel(panel['label'], fontsize=9)
    return mesh


def _prepare(out):
    """Make the parent directory of an output path, so --out can organise.

    Figures belong under analysis/figures/, filed by case and initial time, and
    requiring the directory to exist first turns every plotting command into two.
    """
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return out


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    init = datetime.datetime.strptime(args.init, "%Y%m%d%H")
    runs = [r.split('=', 1) for r in args.run]
    metas = [read_meta(path) for _, path in runs]

    leads = available_leads(runs[0][1])
    if args.lead is None:
        print(f"forecast hours present in {runs[0][0]}: "
              + ", ".join(str(h) for h in leads))
        return
    if args.lead not in leads:
        raise SystemExit(f"+{args.lead} h is not in this run; available: {leads}")

    valid = init + datetime.timedelta(hours=args.lead)
    columns = [(name, load_run(path, args.lead, meta))
               for (name, path), meta in zip(runs, metas)]
    if args.era5:
        n = columns[0][1].sfc.shape[0]
        columns.insert(0, ("ERA5", load_era5(args.era5, args.tc_id, valid, n,
                                             metas[0])))

    panels = [PANELS[i] for i in args.panels] if args.panels else PANELS[:N_DEFAULT]
    nrow, ncol = len(panels), len(columns)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow),
                             squeeze=False)

    for r, panel in enumerate(panels):
        # One colour scale per row, so columns are actually comparable. Using
        # each panel's own range would make a weaker forecast look identical to
        # a stronger one.
        vals = np.concatenate([mask_outside(f, panel['shade'](f), args.mask_radius).ravel()
                               for _, f in columns])
        lo, hi = (float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99)))
        if panel.get('sym'):
            hi = max(hi, abs(lo)); lo = -hi
        vlim = (lo, hi)
        # One arrow scale per row, for the same reason as the colour scale.
        speeds = np.concatenate([
            np.hypot(*[mask_outside(f, a, args.mask_radius)
                       for a in panel['wind'][1](f)]).ravel() for _, f in columns])
        qscale = max(float(np.nanpercentile(speeds, 98)), 1e-6) * 20

        for c, (name, f) in enumerate(columns):
            ax = axes[r][c]
            mesh = draw(fig, ax, f, panel, args.mask_radius, vlim, c == 0, qscale)
            if r == 0:
                ax.set_title(name, fontsize=11)
            if c == ncol - 1:
                cb = fig.colorbar(mesh, ax=axes[r], fraction=0.025, pad=0.01)
                cb.set_label(panel['unit'], fontsize=8)
            if r != nrow - 1:
                ax.set_xticklabels([])

    fig.suptitle(f"{args.tc_id}  init {init:%Y-%m-%d %H}Z  "
                 f"+{args.lead:03d} h  valid {valid:%Y-%m-%d %H}Z", fontsize=12)
    fig.savefig(_prepare(args.out), dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"wrote {args.out}  ({nrow} rows x {ncol} columns)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                   help="a start_from_* directory; repeat to add columns")
    p.add_argument("--era5", default=None,
                   help="directory of {TC_ID}_{time}_combined.nc, adds a truth column")
    p.add_argument("--tc-id", required=True)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--lead", type=int, default=None,
                   help="forecast hour; omit to list what the run contains")
    p.add_argument("--out", default="forecast.png")
    p.add_argument("--panels", type=int, nargs="*", default=None,
                   help=f"subset of rows 0..{len(PANELS)-1}; default is "
                        f"0..{N_DEFAULT-1}. Rows {N_DEFAULT}+ are the "
                        f"diagnostic ones: "
                        + ", ".join(f"{i}={PANELS[i]['label']}"
                                    for i in range(N_DEFAULT, len(PANELS))))
    p.add_argument("--mask-radius", type=float, default=0.0,
                   help="blank beyond this radius in degrees; 0 (the default) "
                        "shows everything the model produced. A default that "
                        "hid the outermost ring - the least constrained part of "
                        "the polar grid - would make the model look better than "
                        "it is, so trimming is opt-in: 9.5 for presentation")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
