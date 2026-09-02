"""Is the forecast vortex too axisymmetric, and does the feedback make it worse?

A tropical cyclone is not a circle. Its asymmetry is what produces beta drift,
what lets a trough ventilate or shear it, and what decides whether it recurves;
a perfectly axisymmetric vortex is a billiard ball that only goes where the mean
flow pushes it. So "the polar model loses asymmetry" is a claim about the
mechanism behind a track error, not a cosmetic observation - and it has been
asserted in this project from a single case at a single lead, with no tool that
could reproduce the number. This is that tool, over a season.

**What is measured.** On a ring of radius r around the storm centre, sample the
field at Ntheta azimuths and remove the ring mean. What is left is the
asymmetry. Two numbers come out of it:

    std    the standard deviation around the ring - total asymmetry
    wn1    the amplitude of azimuthal wavenumber 1 - the component that
           actually moves a vortex

**Why a ratio.** Absolute asymmetry says as much about the storm as about the
model: a system embedded in a sheared environment is asymmetric whatever the
forecast does. Every value is divided by ERA5's at the same case, lead and
radius, so 1.0 means "as asymmetric as the truth" and 0.2 means "one fifth of
it". Cases where the truth's own asymmetry is negligible are dropped rather
than allowed to produce an enormous ratio - see --min-truth.

**Why sampling and not a grid transform.** Both the forecast and the ERA5 box
are storm-following, so the storm sits at the centre of each array by
construction and a ring is well defined in both without regridding. Bilinear
interpolation onto the ring is the only resampling, and it is identical for
every run, so it cannot favour one.

**Radii.** Default 2, 4 and 8 degrees: the inner core, the region the polar
grid resolves best, and the outer vortex where the environment takes over. The
polar runs are NaN outside their 10-degree disc, so radii beyond about 9 have
nothing to read.

Usage
-----
    python tools/season_asymmetry.py --era5-root /wk2/yungyun/FCNV2_TC \\
      --ibtracs /home/payne/ibtracs/ibtracs.WP.list.v04r01.csv \\
      --clip-to-best-track \\
      --runs "cart_1way=/wk2/yungyun/FCNV2_TC@one_way_couple_model" \\
      --runs "polar_1way=/home/payne/LLAT_polar_runs_r80long_full@one-way" \\
      --max-lead 192 --every 24
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_here, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("season_stats")
pf = ss.pf


def clip_times(args, tc_id, init_str, run_dir):
    """Valid times with an IBTrACS record, or None when not clipping."""
    if not (args.clip_to_best_track or args.min_vmax):
        return None
    sid = ss.storm_sid(args.ibtracs, tc_id, init_str, run_dir)
    return (ss.storm_times(args.ibtracs, sid, args.min_vmax)
            if sid else set())


def ring(field2d, r_deg, res_deg, n_theta):
    """Bilinear samples of a storm-centred array on a ring, or None.

    Returns None if any sample lands on a NaN. That is deliberate: the polar
    runs are NaN outside their disc and near the pole of the transform, and a
    ring with holes in it has a different mean and a different variance from a
    complete one, so a partial ring is not comparable with a full one.
    """
    n = field2d.shape[0]
    c = (n - 1) / 2.0
    rr = r_deg / res_deg
    th = np.arange(n_theta) * (2 * np.pi / n_theta)
    y = c + rr * np.sin(th)
    x = c + rr * np.cos(th)
    if x.min() < 0 or y.min() < 0 or x.max() > n - 1.001 or y.max() > n - 1.001:
        return None
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    v = (field2d[y0, x0] * (1 - fx) * (1 - fy)
         + field2d[y0, x0 + 1] * fx * (1 - fy)
         + field2d[y0 + 1, x0] * (1 - fx) * fy
         + field2d[y0 + 1, x0 + 1] * fx * fy)
    return None if not np.isfinite(v).all() else v


def asymmetry(v):
    """(std, wavenumber-1 amplitude) of a ring, after removing its mean."""
    a = v - v.mean()
    n = len(a)
    # Amplitude of the wn-1 component: 2|F1|/n, so it is half the peak-to-peak
    # of a pure sinusoid and directly comparable with the std of one.
    f1 = np.fft.rfft(a)[1]
    return float(a.std()), float(2.0 * abs(f1) / n)


def collect(run_dir, era5_dir, tc_id, init_str, args, truth_cache):
    """{(lead, radius): (std_ratio, wn1_ratio)} for one case."""
    init = datetime.datetime.strptime(init_str, "%Y%m%d%H")
    meta = pf.read_meta(run_dir)
    clip = clip_times(args, tc_id, init_str, run_dir)
    out = {}
    for h in pf.available_leads(run_dir):
        if h > args.max_lead or h % args.every:
            continue
        valid = init + datetime.timedelta(hours=h)
        if clip is not None and valid not in clip:
            continue
        try:
            f = pf.load_run(run_dir, h, meta, upper=False)
            n = f.lon.shape[0]
            key = (tc_id, valid)
            if key not in truth_cache:
                t = pf.load_era5(era5_dir, tc_id, valid, n, meta, upper=False)
                res = abs(float(t.lon[0, 1] - t.lon[0, 0]))
                truth_cache[key] = {
                    r: (asymmetry(rv) if rv is not None else None)
                    for r in args.radii
                    for rv in [ring(np.asarray(t.s(args.var), float), r, res,
                                    args.n_theta)]}
            truth = truth_cache[key]
        except (FileNotFoundError, OSError, KeyError):
            continue
        res = abs(float(f.lon[0, 1] - f.lon[0, 0]))
        a = np.asarray(f.s(args.var), dtype=float)
        for r in args.radii:
            tr = truth.get(r)
            if tr is None or tr[0] < args.min_truth:
                continue          # the truth is round here; a ratio is noise
            fv = ring(a, r, res, args.n_theta)
            if fv is None:
                continue
            fs, fw = asymmetry(fv)
            out[(h, r)] = (fs / tr[0], fw / tr[1] if tr[1] > 0 else np.nan)
    return out


def main(args):
    args.radii = [float(x) for x in args.radii.split(",")]
    runs = [ss.parse_run(s, args.mode) for s in args.runs]
    if (args.clip_to_best_track or args.min_vmax) and not args.ibtracs:
        raise SystemExit("--clip-to-best-track and --min-vmax need --ibtracs")
    if args.ibtracs:
        print(f"IBTrACS: {len(ss.ibt.load(args.ibtracs))} storms", flush=True)

    found = {}
    for label, root, mode, subdir in runs:
        found[label] = {(tc, i): p for tc, i, p in
                        ss.find_starts(root, args.version, mode, subdir)}
        print(f"  {label}: {len(found[label])} cases")
    common = sorted(set.intersection(*(set(v) for v in found.values())))
    print(f"comparing on the {len(common)} initial times every run has",
          flush=True)

    # truth is the same field for every run, so it is computed once per case
    acc = {l: {} for l, *_ in runs}
    truth_cache = {}
    for i, (tc, init) in enumerate(common):
        if args.limit and i >= args.limit:
            break
        era5 = os.path.join(os.path.expanduser(args.era5_root), tc, 'ERA5',
                            'for_DLAMPty')
        for label, *_ in runs:
            for k, v in collect(found[label][(tc, init)], era5, tc, init,
                                args, truth_cache).items():
                acc[label].setdefault(k, []).append(v)
        if args.print_every and (i + 1) % args.print_every == 0:
            print(f"    {i + 1}/{len(common)}", flush=True)

    leads = sorted({h for l in acc for (h, _r) in acc[l]})
    for r in args.radii:
        print(f"\n===== {args.var}, ring at r = {r:g} deg =====")
        print("  median asymmetry as a fraction of ERA5's; "
              "below 1 is too axisymmetric")
        print(f"{'run':<12}{'metric':>7}" + "".join(f"{h:>9.0f}h" for h in leads))
        print("-" * (19 + 10 * len(leads)))
        for label, *_ in runs:
            for j, mname in enumerate(("std", "wn1")):
                line = f"{label:<12}{mname:>7}"
                for h in leads:
                    vals = [v[j] for v in acc[label].get((h, r), [])
                            if np.isfinite(v[j])]
                    line += (f"{np.median(vals):>10.2f}" if len(vals) >= args.min_n
                             else f"{'-':>10}")
                print(line)
        line = f"{'(n)':<12}{'':>7}"
        for h in leads:
            line += f"{len(acc[runs[0][0]].get((h, r), [])):>10}"
        print(line)

    draw(acc, runs, leads, args)


def draw(acc, runs, leads, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(args.radii),
                             figsize=(4.6 * len(args.radii), 7.4),
                             squeeze=False, sharex=True)
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for col, r in enumerate(args.radii):
        for j, mname in enumerate(("std", "wn1")):
            ax = axes[j][col]
            for (label, *_), c in zip(runs, colours * 4):
                xs, ys = [], []
                for h in leads:
                    vals = [v[j] for v in acc[label].get((h, r), [])
                            if np.isfinite(v[j])]
                    if len(vals) >= args.min_n:
                        xs.append(h)
                        ys.append(float(np.median(vals)))
                if xs:
                    ax.plot(xs, ys, 'o-', ms=4, lw=1.8, color=c, label=label)
            ax.axhline(1.0, color='0.3', lw=1)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_title(f"r = {r:g} deg", fontsize=11)
            else:
                ax.set_xlabel("forecast hour")
            if col == 0:
                ax.set_ylabel(f"{mname} / ERA5")
            ax.legend(fontsize=8)
    fig.suptitle(f"Azimuthal asymmetry of {args.var} relative to ERA5\n"
                 "1.0 is as asymmetric as the truth; below it the vortex is "
                 "too round, and a round vortex does not drift", fontsize=11)
    fig.tight_layout()
    out = os.path.expanduser(args.out or os.path.join(
        "analysis", "figures", "season",
        "asymmetry_" + "_".join(l.replace('/', '-') for l, *_ in runs) + ".png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    print("\nwrote " + out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True,
                   metavar="NAME=PATH[@SELECTOR]")
    p.add_argument("--era5-root", required=True)
    p.add_argument("--ibtracs", default=None)
    p.add_argument("--min-vmax", type=float, default=0.0,
                   help="score only leads where the best-track Vmax at that "
                        "time is at least this, in kt. 65 is the paper's TY "
                        "group. This is the intensity AT THE FORECAST TIME, "
                        "not the storm's lifetime peak: a typhoon that has "
                        "decayed to a depression contributes its strong hours "
                        "and not its weak ones, which is what separates "
                        "'the model is bad at typhoons' from 'the sample is "
                        "mostly weak systems'. Implies --clip-to-best-track")
    p.add_argument("--clip-to-best-track", action="store_true")
    p.add_argument("--version", default=None)
    p.add_argument("--mode", default="one-way",
                   choices=["one-way", "two-way", "standalone"])
    p.add_argument("--var", default="msl",
                   help="surface field to measure; msl is the vortex's own "
                        "signature, u10 or v10 measure the flow asymmetry")
    p.add_argument("--radii", default="2,4,8",
                   help="degrees from the centre. The polar runs are NaN "
                        "outside 10 deg, so nothing beyond about 9 can be read")
    p.add_argument("--n-theta", type=int, default=72,
                   help="samples around the ring. 72 is every 5 degrees, far "
                        "more than the 0.25 deg grid can resolve at r=2 and "
                        "cheap enough not to matter")
    p.add_argument("--min-truth", type=float, default=20.0,
                   help="skip a ring where ERA5's own asymmetry is below this, "
                        "in the field's units (Pa for msl). Dividing by a "
                        "number near zero manufactures huge ratios out of a "
                        "storm that simply was round")
    p.add_argument("--min-n", type=int, default=10,
                   help="cells with fewer samples print blank")
    p.add_argument("--max-lead", type=float, default=192)
    p.add_argument("--every", type=int, default=24)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
