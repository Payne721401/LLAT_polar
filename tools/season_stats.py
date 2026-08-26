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

# (tc_id, valid_time) -> centre. Starts overlap, so the +24 h truth of one is the
# +0 h truth of the next, and without this each file is opened a dozen times from
# a network filesystem. That is what makes a season look like it has hung.
_TRUTH = {}


def truth_centre(era5_dir, tc_id, valid, n, meta):
    key = (tc_id, valid)
    if key not in _TRUTH:
        _TRUTH[key] = te.centre(pf.load_era5(era5_dir, tc_id, valid, n, meta))
    return _TRUTH[key]


def forecast_centre(run_dir, lead, meta):
    """The declared centre, reading only what it needs.

    pf.load_run also loads output_upper_*.npy, which is 2.5 MB against the
    surface file's 0.5 MB and is not consulted here at all. Over a season that is
    several gigabytes pulled across a network filesystem to be discarded, and it
    is why these tools appeared to hang while a single case felt instant.

    mmap_mode leaves the file on disk and touches only the pages holding the last
    two channels, so even the surface array is not read in full.
    """
    p = os.path.join(pf.forecast_dir(run_dir), f"output_sfc_{lead:0>3}h.npy")
    sfc = np.load(p, mmap_mode='r')
    names = meta['surface_vars'] if meta else pf.SFC
    lon = np.asarray(sfc[..., names.index('lon')], dtype=float)
    lat = np.asarray(sfc[..., names.index('lat')], dtype=float)
    return float(np.nanmean(lon)), float(np.nanmean(lat)), sfc.shape[0]


# Prefix only. The model version is part of the directory name, and --version is
# a single flag for every --runs, so requiring an exact match made the one thing
# this tool exists for - a retrained model against its baseline - impossible:
# the two live under different version directories by construction. Globbing the
# prefix finds whichever is there, and the resolved name is printed so a wrong
# directory is visible rather than silently empty.
# No underscore before the star. The polar runs name themselves
# `one_way_couple_model_LLAT_polar_p1_v1`, but the Cartesian runs already sitting
# in /wk2/yungyun/FCNV2_TC/{TC_ID}/ are plain `one_way_couple_model` with no
# version suffix - and `one_way_couple_model_*` does not match that, so the whole
# Cartesian control read as an empty run despite having the same layout and the
# same output_sfc_NNNh.npy files. One character.
MODE_DIR = {'one-way': 'one_way_couple_model*',
            'two-way': '2_way_circle_couple_model*',
            'standalone': 'standalone*'}

# Directories the prefix glob would sweep in but which are separate experiments.
# `_scale1.45` and friends are --frame-speed-scale runs: the same model with the
# frame displacement multiplied, so counting them as the baseline mixes a
# post-processed forecast into the thing it was measured against.
EXCLUDE = ('_scale',)


def find_starts(root, version, mode):
    """Every (tc_id, init) a sweep produced, from the directory layout alone."""
    sub = MODE_DIR[mode] if not version else MODE_DIR[mode].replace('*', version)
    pattern = os.path.join(os.path.expanduser(root), '*', sub, 'start_from_*')
    out = []
    for p in sorted(glob.glob(pattern)):
        m = re.search(r'start_from_(\d{10})$', p)
        if not m or any(x in p.split(os.sep)[-2] for x in EXCLUDE):
            continue
        out.append((p.split(os.sep)[-3], m.group(1), p))
    return out


def errors_for(run_dir, era5_dir, tc_id, init_str):
    """Position error at each lead, or an empty dict if truth is unavailable."""
    init = datetime.datetime.strptime(init_str, "%Y%m%d%H")
    meta = pf.read_meta(run_dir)
    out = {}
    for h in pf.available_leads(run_dir):
        try:
            flon, flat, n = forecast_centre(run_dir, h, meta)
            tlon, tlat = truth_centre(era5_dir, tc_id,
                                      init + datetime.timedelta(hours=h),
                                      n, meta)
        except (FileNotFoundError, OSError):
            continue                      # ERA5 is 6-hourly; storms also end
        ex, ey = te.km(flon - tlon, flat - tlat, tlat)
        out[h] = float(np.hypot(ex, ey))
    return out


def collect(root, era5_root, version, mode, limit=0, keep=None):
    """Every case's error curve, keyed by lead, plus the per-case curves."""
    by_lead, per_case, cases, skipped = {}, [], 0, []
    starts = find_starts(root, version, mode)
    if keep is not None:
        starts = [s for s in starts if (s[0], s[1]) in keep]
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
        per_case.append((tc, init, path, e))
        for h, v in e.items():
            by_lead.setdefault(h, []).append(v)
    return by_lead, cases, skipped, per_case


def head_to_head(all_cases, runs, args):
    """Which cases the second run wins and loses, against the first.

    A median says the polar model is 8 % behind; it cannot say whether that is
    every case slightly behind or a handful catastrophically so, and those call
    for different work. Ranking each case by the DIFFERENCE separates them, and
    the two tails are the cases worth plotting: one shows what the geometry
    buys, the other what it costs.

    Ranked at a single lead, because a case can win at 24 h and lose at 120 and
    a rank over all leads at once would average that away.
    """
    if len(runs) < 2:
        return
    base, other = runs[0][0], runs[1][0]
    h = args.worst_at
    rows = []
    for key in set(all_cases[base]) & set(all_cases[other]):
        a = all_cases[base][key].get(h)
        b = all_cases[other][key].get(h)
        if a is None or b is None:
            continue
        rows.append((b - a, a, b, key))
    if not rows:
        print("\nno case has both runs at +" + str(h) + " h")
        return
    rows.sort()
    n = max(1, min(args.worst or 10, len(rows) // 2))
    wins = sum(1 for r in rows if r[0] < 0)
    print("\n===== " + other + " against " + base + " at +" + str(h) +
          " h, " + str(len(rows)) + " cases =====")
    print("  " + other + " is closer in " + str(wins) + " of " + str(len(rows)) +
          " (" + format(100.0 * wins / len(rows), ".0f") + " %)")
    for title, sel in (("BEST for " + other, rows[:n]),
                       ("WORST for " + other, rows[-n:][::-1])):
        print("\n  " + title)
        print("  {:<10}{:<12}{:>12}{:>12}{:>10}".format(
            "storm", "init", base[:11], other[:11], "diff"))
        for d, a, b, (tc, init) in sel:
            print("  {:<10}{:<12}{:>11.0f}k{:>11.0f}k{:>+9.0f}k".format(
                tc, init, a, b, d))
    if args.csv:
        out = os.path.expanduser(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("run,tc,init,lead_h,error_km\n")
            for name in all_cases:
                for (tc, init), e in sorted(all_cases[name].items()):
                    for lead in sorted(e):
                        fh.write("{},{},{},{},{:.2f}\n".format(
                            name, tc, init, lead, e[lead]))
        print("\nwrote " + out)


def report_worst(per_case, era5_root, n_worst, at_lead=None):
    """Name the worst cases, with a command that plots each one.

    Finding them by hand means reading a table, picking a storm, then assembling
    a path out of three conventions - which is how a command full of {TCID}
    placeholders gets run verbatim.
    """
    scored = []
    for tc, init, path, e in per_case:
        if not e:
            continue
        h = at_lead if at_lead in e else max(e)
        scored.append((e[h], h, tc, init, path))
    scored.sort(reverse=True)

    print(f"\nworst {min(n_worst, len(scored))} cases"
          + (f" at +{at_lead} h" if at_lead else " at their last common lead"))
    for err, h, tc, init, path in scored[:n_worst]:
        print(f"\n  {tc} {init}   {err:.0f} km at +{h} h")
        print(f"    python tools/track_error.py \\\n"
              f"      --run \"{tc}={path}\" \\\n"
              f"      --era5 {os.path.join(era5_root, tc, 'ERA5', 'for_DLAMPty')} \\\n"
              f"      --tc-id {tc} --init {init} \\\n"
              f"      --out analysis/figures/forecasts/{tc}/{init}/track.png")


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [r.split('=', 1) if '=' in r else (args.mode, r) for r in args.runs]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Restrict every run to the cases they all have. A median over 86 cases and a
    # median over 82 different ones differ for two reasons at once, and the one
    # being asked about is not separable from the other.
    keep = None
    if len(runs) > 1 and not args.unpaired:
        for _, root in runs:
            ids = {(tc, init) for tc, init, _ in
                   find_starts(root, args.version, args.mode)}
            keep = ids if keep is None else (keep & ids)
        print(f"comparing on the {len(keep)} initial times every run has; "
              f"pass --unpaired to use each run's full set", flush=True)

    all_cases = {}
    for name, root in runs:
        by_lead, n_cases, skipped, per_case = collect(
            root, args.era5_root, args.version, args.mode, args.limit, keep)
        seen = sorted({p.split(os.sep)[-2] for _, _, p, _ in per_case})
        print(f"  {name}: {n_cases} cases under {', '.join(seen) or '(none)'}",
              flush=True)
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

        all_cases[name] = {(tc, init): e for tc, init, _p, e in per_case}

        if args.worst:
            report_worst(per_case, os.path.expanduser(args.era5_root),
                         args.worst, args.worst_at)

    head_to_head(all_cases, runs, args)

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
    p.add_argument("--version", default=None,
                   help="restrict to one model version; by default any version "
                        "directory under --runs is taken, which is what lets a "
                        "retrained model be compared against its baseline")
    p.add_argument("--mode", default="one-way",
                   choices=["one-way", "two-way", "standalone"])
    p.add_argument("--limit", type=int, default=0, help="first N cases, for a smoke test")
    p.add_argument("--unpaired", action="store_true",
                   help="let each run use its own case set instead of the "
                        "intersection; the medians then differ for two reasons")
    p.add_argument("--worst", type=int, default=0,
                   help="also name the N worst cases, each with the command "
                        "that plots it")
    p.add_argument("--csv", default=None,
                   help="write every case at every lead, so the ranking can be "
                        "redone at another lead without recomputing the season")
    p.add_argument("--worst-at", type=int, default=120,
                   help="the lead to rank them at")
    p.add_argument("--print-every", type=int, default=24)
    p.add_argument("--out", default="season.png")
    p.add_argument("--dpi", type=int, default=150)
    main(p.parse_args())
