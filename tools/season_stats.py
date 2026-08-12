"""Aggregate a whole season, because one case cannot settle anything.

Kong-rey's 296 km at +24 h might be the model or might be Kong-rey, and after a
retrain nothing would distinguish "the change helped" from "the next storm was
easier". A season of twenty-odd storms turns a number into a distribution, and a
distribution is what a claim can be tested against.

Reports the median rather than the mean. Track error is strongly skewed - one
storm that misses a recurvature contributes thousands of kilometres and drags a
mean with it - so the median says what a typical forecast does and the
interquartile band says how much that varies. The mean is printed alongside
precisely so the gap between them is visible; where they diverge, a few cases are
carrying the average.

The count at each lead matters too and is printed: storms end, so the sample
shrinks with lead time and the far end of the curve rests on the few long-lived
systems, which are not a random subset.

Usage
-----
    python tools/season_stats.py \\
        --runs ~/LLAT_polar_runs \\
        --era5-root /wk2/yungyun/FCNV2_TC \\
        --version LLAT_polar_vtvr_v1 --mode one-way \\
        --out analysis/figures/forecasts/season_2024.png

--runs may be repeated as NAME=PATH to compare two seasons - a baseline against a
retrained model - on the same axes.
"""
import argparse
import datetime
import glob
import importlib.util
import os
import re

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "track_error", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "track_error.py"))
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)
pf = te.pf

MODE_DIR = {'one-way': 'one_way_couple_model_{v}',
            'two-way': '2_way_circle_couple_model_{v}',
            'standalone': 'standalone_{v}'}


def find_starts(root, version, mode):
    """Every (tc_id, init) a sweep produced, from the directory layout alone."""
    pattern = os.path.join(os.path.expanduser(root), '*',
                           MODE_DIR[mode].format(v=version), 'start_from_*')
    out = []
    for p in sorted(glob.glob(pattern)):
        m = re.search(r'start_from_(\d{10})$', p)
        tc = p.split(os.sep)[-3]
        if m:
            out.append((tc, m.group(1), p))
    return out


def errors_for(run_dir, era5_dir, tc_id, init_str):
    """Position error at each lead, or an empty dict if truth is unavailable."""
    init = datetime.datetime.strptime(init_str, "%Y%m%d%H")
    meta = pf.read_meta(run_dir)
    out = {}
    for h in pf.available_leads(run_dir):
        try:
            f = pf.load_run(run_dir, h, meta)
            t = pf.load_era5(era5_dir, tc_id,
                             init + datetime.timedelta(hours=h),
                             f.sfc.shape[0], meta)
        except (FileNotFoundError, OSError):
            continue                      # ERA5 is 6-hourly; storms also end
        flon, flat = te.centre(f)
        tlon, tlat = te.centre(t)
        ex, ey = te.km(flon - tlon, flat - tlat, tlat)
        out[h] = float(np.hypot(ex, ey))
    return out


def collect(root, era5_root, version, mode, limit=0):
    """Every case's error curve, keyed by lead."""
    by_lead, cases, skipped = {}, 0, []
    starts = find_starts(root, version, mode)
    if limit:
        starts = starts[:limit]
    for tc, init, path in starts:
        era5 = os.path.join(os.path.expanduser(era5_root), tc, 'ERA5',
                            'for_DLAMPty')
        if not os.path.isdir(era5):
            skipped.append(f"{tc} (no ERA5 directory)")
            continue
        e = errors_for(path, era5, tc, init)
        if not e:
            skipped.append(f"{tc}/{init} (no matching truth)")
            continue
        cases += 1
        for h, v in e.items():
            by_lead.setdefault(h, []).append(v)
    return by_lead, cases, skipped


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [r.split('=', 1) if '=' in r else (args.mode, r) for r in args.runs]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for name, root in runs:
        by_lead, n_cases, skipped = collect(root, args.era5_root, args.version,
                                            args.mode, args.limit)
        if not by_lead:
            raise SystemExit(f"{name}: no cases with truth under {root}")

        leads = sorted(by_lead)
        med = np.array([np.median(by_lead[h]) for h in leads])
        mean = np.array([np.mean(by_lead[h]) for h in leads])
        q1 = np.array([np.percentile(by_lead[h], 25) for h in leads])
        q3 = np.array([np.percentile(by_lead[h], 75) for h in leads])
        n = np.array([len(by_lead[h]) for h in leads])

        line, = axes[0].plot(leads, med, 'o-', lw=1.8, ms=3,
                             label=f"{name} (median, {n_cases} cases)")
        c = line.get_color()
        axes[0].fill_between(leads, q1, q3, color=c, alpha=0.18)
        axes[0].plot(leads, mean, '--', lw=1.2, color=c, label=f"{name} (mean)")
        axes[1].plot(leads, n, 'o-', lw=1.5, ms=3, color=c, label=name)

        print(f"\n===== {name} — {n_cases} cases =====")
        if skipped:
            print(f"  skipped {len(skipped)}: {', '.join(skipped[:4])}"
                  + (" ..." if len(skipped) > 4 else ""))
        print(f"{'lead':>5} {'n':>4} {'median':>9} {'mean':>9} "
              f"{'p25':>8} {'p75':>8} {'worst':>9}")
        print("-" * 56)
        for i, h in enumerate(leads):
            if h % args.print_every:
                continue
            print(f"{h:>4}h {n[i]:>4} {med[i]:>8.0f}k {mean[i]:>8.0f}k "
                  f"{q1[i]:>7.0f}k {q3[i]:>7.0f}k {max(by_lead[h]):>8.0f}k")

    axes[0].set_ylabel("position error [km]")
    axes[0].set_title("season median, shaded interquartile range\n"
                      "dashed is the mean — where it separates, a few cases carry it",
                      fontsize=10)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("cases contributing")
    axes[1].set_title("sample size — storms end, and the survivors are not a\n"
                      "random subset of them", fontsize=10)
    axes[1].set_ylim(bottom=0)
    for a in axes:
        a.set_xlabel("forecast hour")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(te._prepare(args.out), dpi=args.dpi, bbox_inches='tight',
                facecolor='white')
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True, metavar="[NAME=]PATH")
    p.add_argument("--era5-root", required=True,
                   help="the directory holding {TC_ID}/ERA5/for_DLAMPty")
    p.add_argument("--version", default="LLAT_polar_vtvr_v1")
    p.add_argument("--mode", default="one-way",
                   choices=["one-way", "two-way", "standalone"])
    p.add_argument("--limit", type=int, default=0, help="first N cases, for a smoke test")
    p.add_argument("--print-every", type=int, default=24)
    p.add_argument("--out", default="season.png")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
