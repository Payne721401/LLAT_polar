"""Is the speed shortfall a fixed fraction or a fixed distance, and does it vary?

Two cases said different things. Kong-rey from 2024-10-25 travelled 516 km in
24 h against an observed 833 - 62 %, short by 317 km. From 2024-10-27 it managed
90 against 339 - 27 %, short by 249. The ratios differ by a factor of 2.3 and the
shortfalls by a quarter, which hints that the error is closer to a fixed distance
than a fixed fraction. Two points cannot settle that; eighty can.

It matters because it decides what the fix looks like. A fixed fraction is a
calibration - multiply the predicted displacement and the problem goes away. A
fixed distance is not: it would mean the model loses roughly the same amount of
motion whatever the storm is doing, and no scalar recovers that, which points at
the objective rather than at post-processing.

Also bins the error by latitude, because the training set is tropical-cyclone
data and the sample thins as storms recurve poleward. A model that does markedly
worse above 25 N would be short of examples there rather than short of physics.

Usage
-----
    python tools/season_bias.py \\
        --runs ~/LLAT_polar_runs --era5-root /wk2/yungyun/FCNV2_TC \\
        --version LLAT_polar_vtvr_v1 --mode one-way \\
        --out analysis/figures/forecasts/season_bias.png
"""
import argparse
import datetime
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "season_stats", os.path.join(_here, "season_stats.py"))
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)
te, pf = ss.te, ss.pf


def displacements(run_dir, era5_dir, tc_id, init_str, window_h=24):
    """(observed, forecast, latitude) for each window_h interval of one case."""
    init = datetime.datetime.strptime(init_str, "%Y%m%d%H")
    meta = pf.read_meta(run_dir)
    fc, tr = {}, {}
    for h in pf.available_leads(run_dir):
        try:
            f = pf.load_run(run_dir, h, meta)
            tr[h] = ss.truth_centre(era5_dir, tc_id,
                                    init + datetime.timedelta(hours=h),
                                    f.sfc.shape[0], meta)
        except (FileNotFoundError, OSError):
            continue
        fc[h] = te.centre(f)

    out = []
    for h in sorted(tr):
        h0 = h - window_h
        if h0 not in tr or h0 not in fc or h not in fc:
            continue
        lat = tr[h][1]
        o = np.hypot(*te.km(tr[h][0] - tr[h0][0], tr[h][1] - tr[h0][1], lat))
        p = np.hypot(*te.km(fc[h][0] - fc[h0][0], fc[h][1] - fc[h0][1], lat))
        out.append((float(o), float(p), float(lat)))
    return out


def fit_both(obs, pred):
    """Compare pred = k*obs against pred = obs - c, by residual spread.

    Two one-parameter models, so the comparison is fair without penalising
    complexity: whichever leaves the smaller residual describes the error better.
    """
    obs, pred = np.asarray(obs), np.asarray(pred)
    k = float(np.sum(obs * pred) / np.sum(obs * obs))       # least squares, no intercept
    c = float(np.mean(obs - pred))                          # mean shortfall
    r_mult = float(np.sqrt(np.mean((pred - k * obs) ** 2)))
    r_add = float(np.sqrt(np.mean((pred - (obs - c)) ** 2)))
    return k, c, r_mult, r_add


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    for tc, init, path in ss.find_starts(args.runs, args.version, args.mode):
        era5 = os.path.join(os.path.expanduser(args.era5_root), tc, 'ERA5',
                            'for_DLAMPty')
        if os.path.isdir(era5):
            rows += displacements(path, era5, tc, init, args.window)
    if len(rows) < 10:
        raise SystemExit(f"only {len(rows)} intervals found; check --runs and --version")

    obs, pred, lat = (np.array(x) for x in zip(*rows))
    k, c, r_mult, r_add = fit_both(obs, pred)

    print(f"{len(rows)} intervals of {args.window} h, from "
          f"{len(set(l for l in lat)) and len(rows)} samples\n")
    print(f"  multiplicative   forecast = {k:.3f} x observed        "
          f"residual {r_mult:6.1f} km")
    print(f"  additive         forecast = observed - {c:.0f} km      "
          f"residual {r_add:6.1f} km")
    better = "additive" if r_add < r_mult else "multiplicative"
    print(f"\n  the {better} model fits better, by "
          f"{abs(r_add - r_mult):.1f} km of residual")
    if abs(r_add - r_mult) < 0.05 * min(r_add, r_mult):
        print("  — but by under 5 %, so this does not distinguish them")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(obs, pred, s=12, alpha=0.35, edgecolors='none')
    x = np.linspace(0, obs.max() * 1.02, 50)
    axes[0].plot(x, x, 'k-', lw=1.2, label='perfect')
    axes[0].plot(x, k * x, '-', lw=1.8, color='tab:orange',
                 label=f'multiplicative, {k:.2f}x  (rms {r_mult:.0f} km)')
    axes[0].plot(x, np.maximum(x - c, 0), '-', lw=1.8, color='tab:green',
                 label=f'additive, −{c:.0f} km  (rms {r_add:.0f} km)')
    axes[0].set_xlabel(f"observed displacement in {args.window} h [km]")
    axes[0].set_ylabel("forecast displacement [km]")
    axes[0].set_title("if the orange line fits, a scalar correction works;\n"
                      "if the green one does, no scalar can", fontsize=10)

    # The ratio, not the absolute shortfall. Storms move faster as they recurve,
    # so if the error is multiplicative then plotting kilometres by latitude just
    # redraws the speed profile and says nothing new. The ratio divides that out:
    # a flat line means one scalar covers every latitude, and a sloping one means
    # the model is differently wrong where its training examples are thinner.
    edges = np.arange(5, 45, 5.0)
    mids, ratio, speed, n = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lat >= lo) & (lat < hi)
        if m.sum() < 5:
            continue
        mids.append(0.5 * (lo + hi))
        ratio.append(float(np.sum(obs[m] * pred[m]) / np.sum(obs[m] * obs[m])))
        speed.append(float(np.median(obs[m])))
        n.append(int(m.sum()))
    axes[1].bar(mids, ratio, width=4.0, color='tab:blue', alpha=0.75)
    axes[1].axhline(k, ls='--', c='tab:orange', lw=1.5,
                    label=f'season fit, {k:.2f}x')
    axes[1].axhline(1.0, c='k', lw=1.0, label='no bias')
    for x_, y_, n_ in zip(mids, ratio, n):
        axes[1].text(x_, y_ + 0.01, f"n={n_}", ha='center', va='bottom', fontsize=7)
    axes[1].set_xlabel("latitude [°N]")
    axes[1].set_ylabel("forecast / observed displacement")
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(fontsize=8)
    axes[1].set_title("flat means one scalar covers every latitude;\n"
                      "sloping means the bias itself changes", fontsize=10)

    print(f"\n{'latitude':>12} {'n':>5} {'ratio':>8} {'median obs':>12}")
    for x_, r_, s_, n_ in zip(mids, ratio, speed, n):
        print(f"{x_-2.5:>5.0f}-{x_+2.5:<5.0f} {n_:>5} {r_:>8.2f} {s_:>9.0f} km")

    for a in axes:
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(te._prepare(args.out), dpi=args.dpi, bbox_inches='tight',
                facecolor='white')
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--era5-root", required=True)
    p.add_argument("--version", default="LLAT_polar_vtvr_v1")
    p.add_argument("--mode", default="one-way",
                   choices=["one-way", "two-way", "standalone"])
    p.add_argument("--window", type=int, default=24,
                   help="interval over which displacement is measured")
    p.add_argument("--out", default="season_bias.png")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
