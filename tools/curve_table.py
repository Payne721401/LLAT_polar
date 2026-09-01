"""Print any logged tag across several runs, at the steps they share.

plot_curves draws the picture and train_val_gap delivers a verdict. Neither
prints the numbers, and two questions keep needing them.

**Where did the ordering settle.** A run that leads at 20,000 steps may have
trailed at 900 and vice versa: the higher-resolution probes were AHEAD of the
baseline at step 453 and behind it by 17,705. "Which architecture is better"
therefore depends on where you stopped looking, and the only way to know
whether a ranking is stable is to see the crossings.

**Is it about to blow up.** gradient_2norm is logged every step. With
gradient_clip_val at 2, anything above that is being clipped, and a run that
clips from the first hundred steps is the 2026-08-04 failure repeating - bf16
carries an 8-bit mantissa and the LayerNorms either side of DownSample have the
smallest gradients in the network, so they lose precision first. Seeing that at
step 200 costs nothing; discovering it from a rising training loss costs the
run.

Usage
-----
    # where did the ranking settle
    python tools/curve_table.py --tag val_loss \\
        --run "base=runs/probe_base" --run "p226=runs/probe_patch226" \\
        --run "r160=runs/probe_r160"

    # the first two hundred steps of the gradient norm
    python tools/curve_table.py --tag gradient_2norm --head 40 --flag 2 \\
        --run "combo=runs/probe_combo"

    # the historical resolution ladder
    python tools/curve_table.py --tag val_loss --every 20 \\
        --run "R41=runs/prod_lr5e-5" --run "R40=runs/p1_wide" \\
        --run "T360=runs/p1_theta360" --run "pr2=runs/t360_pr2" \\
        --run "R80=runs/t360_r80"
"""
import argparse
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_val_gap", os.path.join(_here, "train_val_gap.py"))
tvg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tvg)


def main(args):
    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs.append((label or os.path.basename(spec.rstrip("/")),
                     os.path.expanduser(path or spec)))

    series = {}
    for label, path in runs:
        merged, _files = tvg.curves(path)
        if args.tag not in merged:
            near = [t for t in merged if args.tag.split("_")[0] in t]
            raise SystemExit(
                f"{label}: no tag {args.tag!r}. {len(merged)} tags logged"
                + (f"; did you mean one of {near[:6]}?" if near else ""))
        series[label] = merged[args.tag]

    steps = sorted(set.intersection(*(set(s) for s in series.values())))
    if not steps:
        raise SystemExit(
            f"the runs share no step at which {args.tag!r} was logged - they "
            f"were probably logged on different intervals. Compare them one at "
            f"a time, or pick a tag they both write at the same cadence.")
    if args.max_step:
        steps = [s for s in steps if s <= args.max_step]
    if args.head:
        steps = steps[:args.head]
    elif args.every > 1:
        # keep the last one whatever the stride, so the end of the run is shown
        steps = steps[::args.every] + ([steps[-1]] if steps[-1] not in
                                       steps[::args.every] else [])

    print(f"{args.tag}   {len(steps)} of the "
          f"{len(set.intersection(*(set(s) for s in series.values())))} shared "
          f"steps")
    head = f"{'step':>8}" + "".join(f"{l:>12}" for l, _ in runs)
    print(head + ("      best" if len(runs) > 1 else ""))
    print("-" * (len(head) + (10 if len(runs) > 1 else 0)))
    lead_runs = []
    for s in steps:
        row = {l: series[l][s] for l, _ in runs}
        best = min(row, key=row.get)
        lead_runs.append(best)
        line = f"{s:>8}"
        for l, _ in runs:
            v = row[l]
            mark = "*" if args.flag and v > args.flag else " "
            line += f"{v:>11.5g}{mark}"
        print(line + (f"   {best}" if len(runs) > 1 else ""))

    if args.flag:
        n = sum(1 for s in steps for l, _ in runs if series[l][s] > args.flag)
        print(f"\n  * above {args.flag:g}: {n} of {len(steps) * len(runs)} "
              f"values. With gradient_clip_val at 2 those steps were clipped;")
        print("  a run clipping from the start is not training, it is being "
              "held together by the clip.")

    if len(runs) > 1:
        # Where the ranking last changed hands - the honest answer to "is this
        # ordering stable or did I just stop at a flattering step".
        flips = [(steps[i], lead_runs[i - 1], lead_runs[i])
                 for i in range(1, len(lead_runs))
                 if lead_runs[i] != lead_runs[i - 1]]
        print()
        if not flips:
            print(f"  {lead_runs[0]} led at every step shown - the ordering "
                  f"never changed hands.")
        else:
            print(f"  the lead changed hands {len(flips)} times; the last was "
                  f"at step {flips[-1][0]}, {flips[-1][1]} -> {flips[-1][2]}.")
            print(f"  Everything before that is a different answer to the same "
                  f"question, so a ranking read")
            print(f"  from a run stopped earlier than {flips[-1][0]} would have "
                  f"been the wrong one.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   metavar="LABEL=DIR")
    p.add_argument("--tag", default="val_loss",
                   help="any logged tag: val_loss, train_loss_epoch, "
                        "gradient_2norm, or a per-layer grad norm")
    p.add_argument("--head", type=int, default=0,
                   help="only the first N shared steps - for watching the "
                        "start of a run")
    p.add_argument("--every", type=int, default=1,
                   help="print every Nth shared step; the last is always kept")
    p.add_argument("--max-step", type=int, default=0)
    p.add_argument("--flag", type=float, default=0.0,
                   help="mark values above this with *. For gradient_2norm use "
                        "the trainer's gradient_clip_val, which is 2 here")
    main(p.parse_args())
