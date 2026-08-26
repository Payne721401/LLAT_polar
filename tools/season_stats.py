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
_BT = {}


def truth_centre(era5_dir, tc_id, valid, n):
    """Where the truth says the storm is, reading only the coordinate arrays.

    Misses are cached as well as hits. ERA5 stops when the storm does while the
    forecast runs on, so before this every case of the same storm retried the
    same absent file at every lead past the end - one filesystem round trip
    each, on NFS, for an answer already known.
    """
    key = (tc_id, valid)
    if key not in _TRUTH:
        try:
            _TRUTH[key] = pf.era5_centre(era5_dir, tc_id, valid, n)
        except (FileNotFoundError, OSError):
            _TRUTH[key] = None
    if _TRUTH[key] is None:
        raise FileNotFoundError(f"no ERA5 truth for {tc_id} at {valid:%Y%m%d%H}")
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


def find_starts(root, version, mode, subdir=None):
    """Every (tc_id, init) a sweep produced, from the directory layout alone.

    Refuses a root where the prefix glob matches more than one model directory.
    /wk2/yungyun/FCNV2_TC holds both 2_way_circle_couple_model and
    2_way_circle_couple_model_v60_e3268 - different models over the same initial
    times - and every caller keys on (tc, init): the dict at the pairing step
    kept whichever sorted last while collect() counted both, so the printed case
    count and the statistics described different sets, and neither said so. A
    wrong model is worse than no answer, so name what is there and let the
    caller choose.

    `subdir` names one model directory outright and skips both the glob and the
    check. `version is None`, not `not version`, so --version "" selects the
    unsuffixed directory; there was no way to ask for it before.
    """
    if subdir is not None:
        sub = subdir
    elif version is None:
        sub = MODE_DIR[mode]
    else:
        sub = MODE_DIR[mode].replace('*', version)
    pattern = os.path.join(os.path.expanduser(root), '*', sub, 'start_from_*')
    out, models = [], {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r'start_from_(\d{10})$', p)
        parts = p.split(os.sep)
        if not m or any(x in parts[-2] for x in EXCLUDE):
            continue
        models[parts[-2]] = models.get(parts[-2], 0) + 1
        out.append((parts[-3], m.group(1), p))
    if len(models) > 1:
        lines = [f'    {v:>4} cases   --runs "NAME={root}@{k}"'
                 for k, v in sorted(models.items())]
        raise SystemExit("\n".join(
            [f"{root}",
             f"  matches {len(models)} model directories under --mode {mode}. "
             f"They are different models",
             f"  over the same initial times, so mixing them is not a "
             f"comparison. Name one:", ""] + lines))
    return out


def errors_for(run_dir, era5_dir, tc_id, init_str, bt=None):
    """Position error at each lead, or an empty dict if truth is unavailable.

    `bt` is a best-track {datetime: (lon, lat)}; without it the truth is ERA5.
    Which one matters less here than it does for intensity - ERA5's position
    sits about 30 km from best track, already ruled out as a source of the
    season's differences - but the paper verifies against IBTrACS, so a number
    quoted beside the paper's should come from the same kind of truth.
    """
    init = datetime.datetime.strptime(init_str, "%Y%m%d%H")
    meta = pf.read_meta(run_dir)
    out = {}
    for h in pf.available_leads(run_dir):
        valid = init + datetime.timedelta(hours=h)
        try:
            flon, flat, n = forecast_centre(run_dir, h, meta)
            if bt is None:
                tlon, tlat = truth_centre(era5_dir, tc_id, valid, n)
            elif valid in bt:
                tlon, tlat = bt[valid]
            else:
                continue          # best track is 6-hourly, the forecast is not
        except (FileNotFoundError, OSError):
            continue                      # ERA5 is 6-hourly; storms also end
        ex, ey = te.km(flon - tlon, flat - tlat, tlat)
        out[h] = float(np.hypot(ex, ey))
    return out


def collect(root, era5_root, version, mode, limit=0, keep=None, subdir=None,
            track_csv=None):
    """Every case's error curve, keyed by lead, plus the per-case curves."""
    by_lead, per_case, cases, skipped = {}, [], 0, []
    starts = find_starts(root, version, mode, subdir)
    if keep is not None:
        starts = [s for s in starts if (s[0], s[1]) in keep]
    if limit:
        starts = starts[:limit]
    for tc, init, path in starts:
        era5 = os.path.join(os.path.expanduser(era5_root), tc, 'ERA5',
                            'for_DLAMPty')
        bt = None
        if track_csv is not None:
            if tc not in _BT:
                _BT[tc] = te.read_best_track(track_csv, tc, "position")
            bt = _BT[tc]
            if not bt:
                skipped.append(f"{tc} (no best-track position)")
                continue
        elif not os.path.isdir(era5):
            skipped.append(f"{tc} (no ERA5 directory)")
            continue
        e = errors_for(path, era5, tc, init, bt)
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


def parse_run(spec, default_mode):
    """"label=root", optionally "label=root@selector" -> (label, root, mode, subdir).

    The selector is either a mode name - one-way, two-way, standalone - or a
    literal model-directory name. The literal form exists because --mode and
    --version are single flags for every --runs, and the comparison this tool is
    for needs neither to be: /wk2/yungyun/FCNV2_TC holds a one-way sweep and two
    different two-way sweeps side by side, so asking for "the Cartesian one-way
    against the Cartesian two-way against the polar one-way" could not be
    written down at all. Naming the directory settles mode and version at once.

        --runs "cart_1way=/wk2/yungyun/FCNV2_TC@one_way_couple_model"
        --runs "cart_2way=/wk2/yungyun/FCNV2_TC@2_way_circle_couple_model"
        --runs "polar_1way=/home/payne/LLAT_polar_runs_r80long_full@one-way"

    The pairing that follows is unchanged, so mixed modes are still compared on
    the initial times every run has.
    """
    label, _, rest = spec.partition('=')
    root, sep, sel = rest.partition('@')
    if not label or not root:
        raise SystemExit(f'--runs {spec!r}: expected "label=path" or '
                         f'"label=path@selector"')
    if not sep:
        return label, root, default_mode, None
    if sel in MODE_DIR:
        return label, root, sel, None
    return label, root, None, sel


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [parse_run(r, args.mode) for r in args.runs]
    track_csv = None
    if args.truth == "best":
        if not args.track_csv:
            raise SystemExit("--truth best needs --track-csv")
        track_csv = os.path.expanduser(args.track_csv)
    print("truth: " + ("best track, " + track_csv if track_csv
                       else "ERA5 domain centre"), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Restrict every run to the cases they all have. A median over 86 cases and a
    # median over 82 different ones differ for two reasons at once, and the one
    # being asked about is not separable from the other.
    keep = None
    if len(runs) > 1 and not args.unpaired:
        for _, root, mode, subdir in runs:
            ids = {(tc, init) for tc, init, _ in
                   find_starts(root, args.version, mode, subdir)}
            keep = ids if keep is None else (keep & ids)
        print(f"comparing on the {len(keep)} initial times every run has; "
              f"pass --unpaired to use each run's full set", flush=True)

    all_cases = {}
    for name, root, mode, subdir in runs:
        by_lead, n_cases, skipped, per_case = collect(
            root, args.era5_root, args.version, mode, args.limit, keep,
            subdir, track_csv)
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
    p.add_argument("--runs", action="append", required=True,
                   metavar="NAME=PATH[@SELECTOR]",
                   help="repeatable. SELECTOR is a mode (one-way, two-way, "
                        "standalone) or a model directory name, which pins "
                        "mode and version together and lets one table mix "
                        "modes; see parse_run")
    p.add_argument("--truth", default="era5", choices=["era5", "best"],
                   help="era5 takes the truth centre from the ERA5 domain's "
                        "coordinates; best reads the best-track position, "
                        "which is what the paper verifies against. The two "
                        "differ by about 30 km, so this changes the numbers "
                        "less than it changes what they can be quoted beside")
    p.add_argument("--track-csv", default=None,
                   help="directory of per-storm best-track CSVs, for "
                        "--truth best")
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
