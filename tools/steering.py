"""Does the model move the storm at the speed its own wind field implies?

The forecast track comes from exactly two of the ninety-eight channels. The grid
follows the storm, so `lon` and `lat` are predicted fields, and the next domain
centre is their mean - nothing else in the system has any say in where the storm
goes. The weather the model produces is never consulted.

That leaves an answerable question. A tropical cyclone is carried by the
environmental flow it is embedded in, so the model's own winds contain a second,
independent statement about how fast the storm should be moving. If the two
statements agree, the coordinate channels are doing their job and the track error
is a genuine forecast error. If the steering flow is right while the coordinate
channels lag, the model already knows the answer and the frame is throwing it
away - which is repairable at inference, without retraining anything.

The diagnostic is the paper's Fig. 15, on forecast output instead of the
idealised experiments: the areal mean wind inside 500 km of the centre at 850,
700, 500 and 300 hPa, the 850-300 hPa deep-layer mean, and the storm's own
translation vector, plus the correlation between the two at each level.

Why an areal mean of the *total* wind is the environmental flow: the vortex
circulation is close to axisymmetric about the centre, so over a disc centred on
it the rotational part very nearly cancels and what survives is the flow the
vortex is sitting in. No vortex removal needed.

Usage
-----
    python tools/steering.py \
        --run "one-way=~/LLAT_polar_runs/.../start_from_2024102500" \
        --era5 /wk2/yungyun/FCNV2_TC/202421W/ERA5/for_DLAMPty \
        --tc-id 202421W --init 2024102500 --out steering.png
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "plot_forecast", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "plot_forecast.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

DEG_KM = 111.32
# The paper's four levels, plus every model level in the 850-300 hPa band for the
# deep-layer mean. 500 km is its radius too; it is wide enough that the vortex
# circulation cancels and narrow enough to still be the flow steering this storm.
PAPER_LEVELS = [850, 700, 500, 300]
STEER_RADIUS_KM = 500.0


def deep_layer(levels):
    """Model levels inside the 850-300 hPa band, deepest first."""
    return sorted((L for L in levels if 300 <= L <= 850), reverse=True)


def disc(field, radius_km):
    """Cells within radius_km of the domain centre."""
    n = field.lon.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    return np.hypot(xx, yy) * res * DEG_KM <= radius_km


def areal_wind(field, level, mask):
    """Mean (u, v) over the disc, in m/s. nanmean: the corners carry no output."""
    return (float(np.nanmean(field.u('u', level)[mask])),
            float(np.nanmean(field.u('v', level)[mask])))


def centre(field):
    return float(np.nanmean(field.lon)), float(np.nanmean(field.lat))


def translation(centres, hours):
    """Storm motion in m/s at each time, by centred difference on the track.

    Centred rather than forward: a forward difference reports the motion over the
    interval *ahead*, which would be compared against a wind field from before it
    and would bias the correlation for no reason.
    """
    out = {}
    for i, h in enumerate(hours):
        a, b = hours[max(i - 1, 0)], hours[min(i + 1, len(hours) - 1)]
        if a == b:
            out[h] = (np.nan, np.nan)
            continue
        lat_ref = centres[h][1]
        dx = (centres[b][0] - centres[a][0]) * DEG_KM * np.cos(np.deg2rad(lat_ref))
        dy = (centres[b][1] - centres[a][1]) * DEG_KM
        dt = (b - a) * 3600.0
        out[h] = (dx * 1000.0 / dt, dy * 1000.0 / dt)
    return out


def collect(loader, hours, levels):
    """Steering wind per level, deep-layer mean, and translation, for one source."""
    centres, wind = {}, {L: {} for L in levels}
    dlm_levels = deep_layer(levels)
    dlm = {}
    for h in hours:
        f = loader(h)
        centres[h] = centre(f)
        m = disc(f, STEER_RADIUS_KM)
        for L in levels:
            wind[L][h] = areal_wind(f, L, m)
        uv = np.array([areal_wind(f, L, m) for L in dlm_levels])
        dlm[h] = (float(uv[:, 0].mean()), float(uv[:, 1].mean()))
    return centres, wind, dlm, translation(centres, hours)


def correlate(flow, motion, hours):
    """Correlation of the u and v components of flow against those of motion.

    Reported per component, as the paper does. A storm steered by the flow gives
    high values in both; a high correlation in one component only usually means
    the track is dominated by a single direction and the other is noise.
    """
    fu = np.array([flow[h][0] for h in hours])
    fv = np.array([flow[h][1] for h in hours])
    mu = np.array([motion[h][0] for h in hours])
    mv = np.array([motion[h][1] for h in hours])
    ok = np.isfinite(fu) & np.isfinite(mu)
    if ok.sum() < 3:
        return float('nan'), float('nan')

    def r(a, b):
        a, b = a[ok], b[ok]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])
    return r(fu, mu), r(fv, mv)


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
    runs = [(n, os.path.expanduser(q))
            for n, q in (r.split('=', 1) for r in args.run)]
    if args.era5:
        args.era5 = os.path.expanduser(args.era5)
    for name, path in runs:
        if not os.path.isdir(os.path.expanduser(path)):
            raise SystemExit(f"run {name!r}: no such directory: {path!r}\n"
                             "(an empty path usually means a shell variable was "
                             "not set in this session)")

    sources = []
    for name, path in runs:
        meta = pf.read_meta(path)
        hours = pf.available_leads(path)
        levels = meta.get('levels', pf.LEVELS)
        sources.append((name, hours, levels,
                        lambda h, p=path, m=meta: pf.load_run(p, h, m)))

    if args.era5:
        if not (args.tc_id and args.init):
            raise SystemExit("--era5 also needs --tc-id and --init")
        name, hours, levels, load = sources[0]
        n = load(hours[0]).sfc.shape[0]
        keep = []
        for h in hours:
            try:
                pf.load_era5(args.era5, args.tc_id,
                             init + datetime.timedelta(hours=h), n, None)
                keep.append(h)
            except FileNotFoundError:
                pass                      # ERA5 is 6-hourly; LLAT steps 3 h
        if keep:
            sources.insert(0, ("ERA5", keep, levels,
                               lambda h, n=n: pf.load_era5(
                                   args.era5, args.tc_id,
                                   init + datetime.timedelta(hours=h), n, None)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    print(f"{'source':<12} {'lead':>5} {'|DLM|':>7} {'|motion|':>9} "
          f"{'ratio':>7} {'turn':>7}")
    print("-" * 52)

    for name, hours, levels, load in sources:
        centres, wind, dlm, motion = collect(load, hours, levels)

        speed_f, speed_m, ratios = [], [], []
        for h in hours:
            sf = float(np.hypot(*dlm[h]))
            sm = float(np.hypot(*motion[h]))
            speed_f.append(sf)
            speed_m.append(sm)
            ratios.append(sm / sf if sf > 1e-6 else np.nan)
            if h % args.print_every == 0:
                # Angle from the steering flow to the motion; a real TC drifts a
                # little left of it in the northern hemisphere (beta drift).
                turn = np.degrees(np.arctan2(motion[h][1], motion[h][0])
                                  - np.arctan2(dlm[h][1], dlm[h][0]))
                turn = (turn + 180) % 360 - 180
                print(f"{name:<12} {h:>4}h {sf:>7.2f} {sm:>9.2f} "
                      f"{ratios[-1]:>7.2f} {turn:>+6.0f}°")

        line, = axes[0].plot(hours, speed_m, 'o-', lw=1.5, ms=3,
                             label=f"{name} motion")
        axes[0].plot(hours, speed_f, '--', lw=1.5, color=line.get_color(),
                     label=f"{name} 850-300 hPa steering")
        axes[1].plot(hours, ratios, 'o-', lw=1.5, ms=3, color=line.get_color(),
                     label=name)

        good = np.isfinite(ratios)
        med = float(np.nanmedian(np.asarray(ratios)[good])) if good.any() else np.nan
        print(f"{'':<12} median motion / steering = {med:.2f}"
              f"   ({len(hours)} times)")
        print(f"{'':<12} correlation of flow with motion, by level:")
        for L in PAPER_LEVELS:
            if L in wind:
                ru, rv = correlate(wind[L], motion, hours)
                print(f"{'':<14} {L:>4} hPa   r(u) {ru:+.2f}   r(v) {rv:+.2f}")
        ru, rv = correlate(dlm, motion, hours)
        print(f"{'':<14} 850-300     r(u) {ru:+.2f}   r(v) {rv:+.2f}\n")

    axes[0].set_ylabel("speed [m s$^{-1}$]")
    axes[0].set_title("storm motion (solid) against the deep-layer steering flow "
                      "it sits in (dashed)", fontsize=10)
    axes[1].axhline(1.0, c='k', lw=0.8)
    axes[1].set_ylabel("motion / steering")
    axes[1].set_xlabel("forecast hour")
    axes[1].set_title("below 1 means the frame is moving slower than the model's "
                      "own winds imply", fontsize=10)
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"{args.tc_id}  init {init:%Y-%m-%d %H}Z  "
                 f"steering within {STEER_RADIUS_KM:.0f} km", fontsize=12)
    fig.tight_layout()
    fig.savefig(_prepare(args.out), dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--era5", default=None,
                   help="adds truth as a reference; without it there is no way "
                        "to tell a weak steering flow from a weak response to it")
    p.add_argument("--tc-id", default=None)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--print-every", type=int, default=24,
                   help="hours between printed rows")
    p.add_argument("--out", default="steering.png")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
