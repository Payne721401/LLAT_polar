"""Does the storm fill after it crosses the coast, and by how much?

A landfalling TC loses its ocean heat source and gains surface friction, so its
central pressure rises fast - tens of hectopascals in a day. A model that keeps
a deep storm over land is missing that, and no track or intensity statistic
averaged over a season will show it, because most cases never make landfall.

This isolates the ones that do. For each case it finds the first lead at which
the truth is over land, then reports the change in central pressure over the
following N hours, for the truth and for each forecast on the same clock.

Truth can be either. ERA5 gives the same MSLP definition the model produces, so
the difference is a like-for-like model error; IBTrACS gives the observed
central pressure, which is what the paper verifies against and what a reader
expects, but it is a different quantity - a best-track estimate against a
0.25-degree cell mean - so the two must not be mixed in one number. Pass
--truth to choose; --truth both prints them side by side and is the honest
default when the gap between them is itself of interest.

Landfall is taken from the model's own landmask channel, recomputed every step
from the storm's position, so "over land" means what the model was told.

Usage
-----
    python tools/landfall.py --era5-root /wk2/yungyun/FCNV2_TC \\
        --track-csv /wk2/yungyun/ERA5_2024_for_TC/TC_list_JMA_v2 \\
        --runs "cartesian=/wk2/yungyun/FCNV2_TC" \\
        --runs "r80_420k=/home/payne/LLAT_polar_runs_r80long_full" \\
        --window 24 --truth both
"""
import argparse
import csv
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


def centre_mslp(field, search_deg=5.0):
    """Minimum MSLP in hPa within search_deg, matching intensity.py."""
    a = np.asarray(field.s("msl"), dtype=float)
    n = a.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    m = (np.hypot(xx, yy) * res <= search_deg) & np.isfinite(a)
    return float(np.nanmin(a[m])) / 100.0 if m.any() else np.nan


def over_land(field, core_deg):
    """Largest landmask value within core_deg of the centre.

    The centre crossing a coast is not the moment a storm starts to feel it -
    a circulation two degrees across is over land well before its eye is - so
    the test is over a disc, not a point.
    """
    lm = np.asarray(field.s("landmask"), dtype=float)
    n = lm.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    res = abs(float(field.lon[0, 1] - field.lon[0, 0]))
    m = (np.hypot(xx, yy) * res <= core_deg) & np.isfinite(lm)
    return float(np.nanmax(lm[m])) if m.any() else 0.0


def read_best_track(csv_dir, tc_id):
    """{datetime: mslp_hPa} from the best-track CSV, if it has a pressure column."""
    p = os.path.join(os.path.expanduser(csv_dir), tc_id + ".csv")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    keys = {k.lower().strip(): k for k in rows[0]}
    tkey = next((keys[k] for k in ("time", "date", "datetime", "yyyymmddhh")
                 if k in keys), None)
    pkey = next((keys[k] for k in ("mslp", "pres", "pressure", "min_pres",
                                   "cp", "slp") if k in keys), None)
    if not tkey or not pkey:
        return {}
    for r in rows:
        raw = str(r[tkey]).strip().replace("-", "").replace(":", "") \
            .replace(" ", "").replace("/", "")
        # Match the format to the string's LENGTH. Trying the longest first
        # and catching ValueError is not enough: strptime accepts a short
        # string against a long format in some builds and returns a silently
        # wrong date - 2024102700 came back as 2024-10-02 07:00.
        fmt = {14: "%Y%m%d%H%M%S", 12: "%Y%m%d%H%M", 10: "%Y%m%d%H",
               8: "%Y%m%d"}.get(len(raw))
        if fmt is None:
            continue
        try:
            t = datetime.datetime.strptime(raw, fmt)
        except ValueError:
            continue
        try:
            out[t] = float(r[pkey])
        except (TypeError, ValueError):
            pass
    return out


def main(args):
    era5_root = os.path.expanduser(args.era5_root)
    runs = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs.append((label or os.path.basename(spec.rstrip("/")), path or spec))

    found = {}
    for label, path in runs:
        found[label] = {(tc, i): p for tc, i, p in
                        ss.find_starts(path, args.version, args.mode)}
        print("  " + label + ": " + str(len(found[label])) + " cases")
    common = sorted(set.intersection(*(set(v) for v in found.values())))
    print("comparing on the " + str(len(common)) + " initial times every run has")

    want_e = args.truth in ("era5", "both")
    want_b = args.truth in ("best", "both") and args.track_csv
    rows = []
    bt_cache = {}
    # Separate reasons. The first version reported "no case had a truth
    # landfall" when in fact every landfall was found and then dropped at the
    # load step, because window was a float and 96 + 24.0 formats as "126.0" in
    # output_sfc_126.0h.npy. A single counter cannot tell those apart.
    n_land, n_short, n_loadfail = 0, 0, 0

    for n, (tc, init_s) in enumerate(common):
        if args.limit and n >= args.limit:
            break
        init = datetime.datetime.strptime(init_s, "%Y%m%d%H")
        era5_dir = os.path.join(era5_root, tc, "ERA5", "for_DLAMPty")
        first = next(iter(found))
        try:
            meta = pf.read_meta(found[first][(tc, init_s)])
            leads = [h for h in pf.available_leads(found[first][(tc, init_s)])
                     if h <= args.max_lead]
        except (FileNotFoundError, OSError):
            continue

        # Find landfall in the TRUTH, so every model is judged on the same
        # clock. A model that never makes landfall would otherwise contribute
        # nothing, which is exactly the failure being looked for.
        t0 = None
        for h in leads:
            try:
                f = pf.load_era5(era5_dir, tc,
                                 init + datetime.timedelta(hours=h), 81, meta)
            except (FileNotFoundError, OSError, KeyError):
                continue
            if over_land(f, args.core_deg) > args.land_threshold:
                t0 = h
                break
        if t0 is None:
            continue
        n_land += 1
        if (t0 + args.window) > max(leads):
            n_short += 1
            continue

        rec = {"tc": tc, "init": init_s, "landfall_h": t0}
        ok = True
        if want_e:
            try:
                a = centre_mslp(pf.load_era5(
                    era5_dir, tc, init + datetime.timedelta(hours=t0), 81, meta),
                    args.search_deg)
                b = centre_mslp(pf.load_era5(
                    era5_dir, tc,
                    init + datetime.timedelta(hours=t0 + args.window), 81, meta),
                    args.search_deg)
                rec["ERA5"] = b - a
            except (FileNotFoundError, OSError, KeyError):
                ok = False
        if want_b:
            if tc not in bt_cache:
                bt_cache[tc] = read_best_track(args.track_csv, tc)
            bt = bt_cache[tc]
            ta = init + datetime.timedelta(hours=t0)
            tb = ta + datetime.timedelta(hours=args.window)
            if ta in bt and tb in bt:
                rec["best track"] = bt[tb] - bt[ta]

        for label, _ in runs:
            try:
                m = pf.read_meta(found[label][(tc, init_s)])
                a = centre_mslp(pf.load_run(found[label][(tc, init_s)], t0, m),
                                args.search_deg)
                b = centre_mslp(pf.load_run(found[label][(tc, init_s)],
                                            t0 + args.window, m),
                                args.search_deg)
                rec[label] = b - a
            except (FileNotFoundError, OSError, KeyError):
                ok = False
        if ok:
            rows.append(rec)
        else:
            n_loadfail += 1
        if args.print_every and (n + 1) % args.print_every == 0:
            print("    " + str(n + 1) + "/" + str(len(common)) +
                  ", " + str(len(rows)) + " landfalls", flush=True)

    print("")
    print(str(n_land) + " cases had a truth landfall; " +
          str(n_short) + " had less than " + str(args.window) +
          " h of forecast after it; " + str(n_loadfail) +
          " failed to load; " + str(len(rows)) + " usable")
    if not rows:
        raise SystemExit(
            "nothing usable. If the landfall count above is non-zero the "
            "failure is in loading, not in finding them - check that "
            "output_sfc_<lead>h.npy exists at the landfall lead plus "
            "--window.")

    cols = (["ERA5"] if want_e else []) + \
           (["best track"] if want_b and any("best track" in r for r in rows) else []) + \
           [l for l, _ in runs]

    print("\nMSLP change over " + str(args.window) +
          " h after the truth's landfall, " + str(len(rows)) + " cases")
    print("positive means filling, which is what a landfalling storm does\n")
    print("{:<14}{:>10}{:>10}{:>10}{:>8}".format(
        "source", "median", "mean", "p25..p75", "n"))
    print("-" * 54)
    for c in cols:
        v = np.array([r[c] for r in rows if c in r and np.isfinite(r[c])])
        if not v.size:
            continue
        print("{:<14}{:>+10.1f}{:>+10.1f}{:>10}{:>8}".format(
            c, np.median(v), v.mean(),
            "{:+.0f}..{:+.0f}".format(np.percentile(v, 25),
                                      np.percentile(v, 75)), v.size))

    print()
    base = "ERA5" if want_e else cols[0]
    bv = {(r["tc"], r["init"]): r[base] for r in rows if base in r}
    for label, _ in runs:
        pairs = [(bv[(r["tc"], r["init"])], r[label]) for r in rows
                 if label in r and (r["tc"], r["init"]) in bv]
        if not pairs:
            continue
        d = np.array([p[1] - p[0] for p in pairs])
        print("{:<14} fills {:+.1f} hPa less than {} on average".format(
            label, -d.mean(), base)
            if d.mean() < 0 else
            "{:<14} fills {:+.1f} hPa more than {} on average".format(
                label, d.mean(), base))
    print("\nA model that fills far less than the truth is not feeling the "
          "land.\nCheck tools/terrain_check.py on the worst of them: it may "
          "simply have\nput the storm somewhere that never had a coast under "
          "it.")

    draw(rows, cols, runs, args)

    if args.csv:
        out = os.path.expanduser(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["tc", "init", "landfall_h"] + cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote " + out)


def draw(rows, cols, runs, args):
    """Three views, because the mean alone hid the result the first time.

    A box plot for the distribution: the polar model's MEDIAN change is
    negative while its mean is positive, which means most cases keep deepening
    and a few that fill hard drag the average up. A mean would have reported
    "fills a little" and been wrong about the typical storm.

    A scatter against truth for the per-case relationship, with the one-to-one
    line: points below it are storms the model failed to fill.

    A histogram of model-minus-truth for how one-sided that failure is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = cols[0]
    labels = [c for c in cols]
    data = [np.array([r[c] for r in rows if c in r and np.isfinite(r[c])])
            for c in labels]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    bp = ax[0].boxplot(data, labels=labels, showmeans=True, widths=0.6)
    ax[0].axhline(0, color="0.4", lw=1, ls="--")
    ax[0].set_ylabel("MSLP change over " + str(args.window) + " h [hPa]")
    ax[0].set_title("after the truth's landfall, " + str(len(rows)) +
                    " cases\n"
                    "positive is filling; the dashed line is no change")
    ax[0].grid(alpha=0.3, axis="y")
    for t in ax[0].get_xticklabels():
        t.set_rotation(20)

    bv = np.array([r[base] for r in rows if base in r])
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (label, _), c in zip(runs, colours):
        v = np.array([r[label] for r in rows if label in r and base in r])
        if v.size != bv.size:
            continue
        ax[1].scatter(bv, v, s=12, alpha=0.5, color=c, label=label)
        ax[2].hist(v - bv, bins=30, histtype="step", lw=1.6, color=c,
                   label=label)
    lim = [min(bv.min(), -20), max(bv.max(), 40)]
    ax[1].plot(lim, lim, "k-", lw=1)
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].axvline(0, color="0.6", lw=0.8)
    ax[1].set_xlabel(base + " change [hPa]")
    ax[1].set_ylabel("forecast change [hPa]")
    ax[1].set_title("per case, against truth\n"
                    "below the line is failing to fill")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    ax[2].axvline(0, color="0.4", lw=1, ls="--")
    ax[2].set_xlabel("forecast minus " + base + " [hPa]")
    ax[2].set_ylabel("cases")
    ax[2].set_title("how one-sided the failure is")
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=8)

    out = os.path.expanduser(args.out or os.path.join(
        "analysis", "figures", "season",
        "landfall_" + "_".join(l.replace("/", "-") for l, _ in runs) +
        "_" + str(args.window) + "h.png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote " + out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True,
                   metavar="[NAME=]PATH")
    p.add_argument("--era5-root", required=True)
    p.add_argument("--track-csv", default=None,
                   help="directory of {TC_ID}.csv, for the best-track column")
    p.add_argument("--truth", default="era5", choices=("era5", "best", "both"))
    p.add_argument("--window", type=int, default=24,
                   help="hours after landfall over which to measure the change")
    p.add_argument("--core-deg", type=float, default=1.0,
                   help="radius searched for land; a circulation feels a coast "
                        "before its centre crosses one")
    p.add_argument("--land-threshold", type=float, default=0.5)
    p.add_argument("--search-deg", type=float, default=5.0)
    p.add_argument("--max-lead", type=int, default=192)
    p.add_argument("--version", default=None)
    p.add_argument("--mode", default="one-way",
                   choices=("one-way", "two-way", "standalone"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--out", default=None,
                   help="figure path; defaults to analysis/figures/season/"
                        "landfall_<labels>_<window>h.png")
    p.add_argument("--csv", default=None)
    main(p.parse_args())
