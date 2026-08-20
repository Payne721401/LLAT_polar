"""Training curves for several runs at once: against steps and against hours.

train_val_gap.py prints the numbers that decide whether a run has room left in
it. This draws them, because the shape of a comparison is what gets read in a
meeting, and because two axes answer two different questions:

  against steps   which architecture learns more per step - the fair comparison
                  between designs, since every run here does the same work per
                  step for its own geometry
  against hours   which architecture is worth the compute - t360 costs 1.18x
                  P1 per step and t360_long twice that again, and a curve that
                  wins on steps can lose on wall clock

Wall clock comes from the wall_time stamped on every TFRecord event, which is
the only elapsed-time signal that reaches the file: train.py logs
elapsed_time_hours with logger=False, so it is never written. A resumed run
shows a jump where the job was queued; that gap is elapsed time but not compute,
so it is reported rather than quietly counted.

The train/val gap is drawn as a third panel, shaded, because that is where
t360_long's story is - a 6.6 % better validation loss bought with a gap that
went from 12.5 % to 22.5 %, which is a different kind of result from P1's, where
the training loss got worse and validation improved anyway.

Adding a run is one more --run. The R experiments slot in with no change:

    python tools/plot_curves.py \\
        --run "P1=runs/p1_wide" \\
        --run "t360=runs/p1_theta360" \\
        --run "t360_long=runs/t360_long" \\
        --run "pr2=runs/t360_pr2" \\
        --run "r80=runs/t360_r80"

Usage
-----
    cd $HOME/LLAT_polar
    python tools/plot_curves.py --run "P1=runs/p1_wide" \\
        --run "t360=runs/p1_theta360" --run "t360_long=runs/t360_long"

Writes analysis/figures/training/curves.png unless --out says otherwise.
"""
import argparse
import importlib.util
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_val_gap", os.path.join(_here, "train_val_gap.py"))
tvg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tvg)


def series(run_dir):
    """Validation, training and elapsed hours for one run, aligned on step."""
    merged, wall, files, idle = tvg.curves_timed(os.path.expanduser(run_dir))
    _vn, val = tvg.pick(merged, tvg.VAL_KEYS)
    _tn, train = tvg.pick(merged, tvg.TRAIN_KEYS)
    if not val:
        raise SystemExit(f"{run_dir}: no validation scalar. tags: "
                         f"{sorted(merged)[:10]}")
    steps = sorted(val)
    tsteps = sorted(train) if train else []

    def nearest(s):
        prev = [t for t in tsteps if t <= s]
        return prev[-1] if prev else (tsteps[0] if tsteps else None)

    v = np.array([val[s] for s in steps])
    t = np.array([train[nearest(s)] for s in steps]) if tsteps else None
    # Hours are stamped per event, not per validation step, so take the nearest
    # stamped step at or before each one.
    wsteps = sorted(wall)
    h = np.array([wall[max([w for w in wsteps if w <= s], default=wsteps[0])]
                  for s in steps]) if wsteps else None
    return np.array(steps, dtype=float), v, t, h, len(files), idle


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = []
    for spec in args.run:
        label, _, path = spec.partition('=')
        runs.append((label or os.path.basename(spec.rstrip('/')), path or spec))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    print(f"{'run':<14}{'best val':>10}{'at step':>10}{'of':>9}{'%':>6}"
          f"{'hours':>8}{'gap':>8}")
    print("-" * 65)
    for (label, path), c in zip(runs, colours):
        steps, v, t, h, nfiles, idle = series(path)
        # Drop the warmup: the first epochs span two thirds of the y-range and
        # flatten everything that matters into a line at the bottom.
        keep = steps >= args.skip * steps.max()
        s, vv = steps[keep], v[keep]
        tt = t[keep] if t is not None else None
        hh = h[keep] if h is not None else None

        ax[0].plot(s, vv, '-', color=c, lw=1.4, label=label)
        if hh is not None:
            ax[1].plot(hh, vv, '-', color=c, lw=1.4, label=label)
        # The classic single-run picture: training and validation on one axis.
        # Off by default because with four runs it doubles the lines and the
        # comparison stops being readable; with one run it is the whole point.
        if args.train and tt is not None:
            ax[0].plot(s, tt, ':', color=c, lw=1.2,
                       label=f"{label} (train)")
            if hh is not None:
                ax[1].plot(hh, tt, ':', color=c, lw=1.2)
        if tt is not None:
            gap = 100.0 * (vv - tt) / tt
            ax[2].plot(s, gap, '-', color=c, lw=1.4, label=label)

        ib = int(np.argmin(v))
        ax[0].plot(steps[ib], v[ib], 'o', color=c, ms=6)
        pct = 100.0 * steps[ib] / steps.max()
        gb = (100.0 * (v[ib] - t[ib]) / t[ib]) if t is not None else np.nan
        total = h[-1] if h is not None else np.nan
        print(f"{label:<14}{v[ib]:>10.5f}{steps[ib]:>10.0f}"
              f"{steps.max():>9.0f}{pct:>6.0f}{total:>8.1f}{gb:>7.1f}%"
              + (f"   (resumed, {idle:.0f} h idle removed)"
                 if idle > 0.5 else ""))

    ax[0].set_xlabel("step"), ax[0].set_ylabel("validation loss")
    ax[0].set_title("validation against steps\nwhich design learns more per step")
    ax[1].set_xlabel("compute hours"), ax[1].set_ylabel("validation loss")
    ax[1].set_title("validation against compute time\n"
                    "queue time between resumed jobs removed")
    ax[2].set_xlabel("step"), ax[2].set_ylabel("(val - train) / train  [%]")
    ax[2].set_title("generalisation gap\nrising means fit is not transferring")
    for a in ax:
        a.grid(alpha=0.3), a.legend(fontsize=8)
    if args.ylim:
        lo, hi = (float(x) for x in args.ylim.split(','))
        ax[0].set_ylim(lo, hi), ax[1].set_ylim(lo, hi)

    out = os.path.expanduser(args.out or os.path.join(
        "analysis", "figures", "training",
        "curves_" + "_".join(l.replace('/', '-') for l, _ in runs) + ".png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")
    print("The middle panel is the one to read for a compute decision: a run "
          "that\nreaches a lower loss in more hours has not necessarily won.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="label=path to a run directory, repeatable")
    p.add_argument("--train", action="store_true",
                   help="also draw the training loss, dotted. With one --run "
                        "this is the usual train-vs-validation picture; with "
                        "several it doubles the lines and hides the comparison, "
                        "which is why it is off by default")
    p.add_argument("--skip", type=float, default=0.05,
                   help="drop this fraction of the start. The warmup spans two "
                        "thirds of the y-range and flattens everything else")
    p.add_argument("--ylim", default=None, metavar="LO,HI",
                   help="e.g. 0.22,0.27 to zoom on the converged region")
    p.add_argument("--out", default=None,
                   help="defaults to analysis/figures/training/curves_<labels>"
                        ".png, named after the runs being compared. A fixed "
                        "default overwrites the previous comparison without "
                        "saying so")
    main(p.parse_args())
