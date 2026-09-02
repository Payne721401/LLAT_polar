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
    args.tag = args.tag.strip()
    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        label, path = label.strip(), path.strip()
        runs.append((label or os.path.basename(spec.rstrip("/")),
                     os.path.expanduser(path or spec)))

    if args.list is not None:
        merged, _files = tvg.curves(runs[0][1])
        pat = '' if args.list == '*' else args.list
        hit = sorted(t for t in merged if pat in t)
        print(f"{len(hit)} of {len(merged)} tags in {runs[0][0]}"
              + (f" matching {pat!r}" if pat else ""))
        for t in hit:
            ks = sorted(merged[t])
            print(f"  {t:<40} {len(ks):>6} points, steps {ks[0]}..{ks[-1]}")
        return

    if args.ratio is not None:
        if len(runs) < 2:
            raise SystemExit("--ratio needs at least two --run")
        cur = {l: tvg.curves(p)[0] for l, p in runs}
        base = runs[0][0]
        tags = sorted(t for t in cur[base] if args.ratio in t)
        if not tags:
            raise SystemExit(f"no tag contains {args.ratio!r}")
        rows = []
        for t in tags:
            vals = {}
            for l, _ in runs:
                d = cur[l].get(t)
                if not d:
                    break
                ks = sorted(d)
                tail = ks[int(len(ks) * (1 - args.tail)):]
                vals[l] = sum(d[k] for k in tail) / len(tail)
            if len(vals) == len(runs) and vals[base]:
                rows.append((vals[runs[1][0]] / vals[base], t, vals))
        rows.sort()
        print(f"mean over the last {100*args.tail:.0f} % of steps, "
              f"{runs[1][0]} / {base}, {len(rows)} tags matching "
              f"{args.ratio!r}")
        print(f"{'ratio':>8}  {'tag':<44}" +
              "".join(f"{l:>12}" for l, _ in runs))
        print("-" * (54 + 12 * len(runs)))
        for r, t, vals in rows:
            print(f"{r:>8.3f}  {t.replace('grad_2.0_norm/', ''):<44}"
                  + "".join(f"{vals[l]:>12.4g}" for l, _ in runs))
        rs = [r for r, _, _ in rows]
        import statistics as st
        print()
        print(f"  median ratio {st.median(rs):.3f} over {len(rs)} tags; "
              f"{sum(1 for r in rs if r < 1)} of them below 1.")
        print("  A ratio well under 1 everywhere means the second run is being")
        print("  asked to change less - which is what a residual connection")
        print("  does when the identity map is already close to the answer.")
        return

    series = {}
    for label, path in runs:
        merged, _files = tvg.curves(path)
        if args.tag not in merged:
            near = [t for t in merged if args.tag.split("_")[0] in t]
            raise SystemExit(
                f"{label}: no tag {args.tag!r}. {len(merged)} tags logged"
                + (f"; did you mean one of {near[:6]}?" if near else ""))
        series[label] = merged[args.tag]

    # Nearest match, not intersection. Requiring an exact shared step looked
    # right and quietly truncated: five 105k-step runs shared only 28 steps and
    # none past 12,711, because a resumed run replays its step counter and the
    # grids drift apart. The table then showed a tenth of the training and
    # said nothing about it. The first run's steps set the grid; every other
    # run answers with its closest logged step, and a run whose nearest point
    # is further than --tol away is left blank rather than faked.
    grid = sorted(series[runs[0][0]])
    if not grid:
        raise SystemExit(f"{runs[0][0]}: {args.tag!r} has no values")
    keys = {l: sorted(series[l]) for l, _ in runs}

    def near(label, s):
        ks = keys[label]
        i = min(range(len(ks)), key=lambda j: abs(ks[j] - s))
        if abs(ks[i] - s) > args.tol:
            return None, None
        return ks[i], series[label][ks[i]]

    steps = grid
    if args.max_step:
        steps = [s for s in steps if s <= args.max_step]
    if args.head:
        steps = steps[:args.head]
    elif args.every > 1:
        # keep the last one whatever the stride, so the end of the run is shown
        steps = steps[::args.every] + ([steps[-1]] if steps[-1] not in
                                       steps[::args.every] else [])

    print(f"{args.tag}   printing {len(steps)} of {len(grid)} steps on "
          f"{runs[0][0]}'s grid, matched to within {args.tol}")
    head = f"{'step':>8}" + "".join(f"{l:>12}" for l, _ in runs)
    print(head + ("      best" if len(runs) > 1 else ""))
    print("-" * (len(head) + (10 if len(runs) > 1 else 0)))
    lead_runs, n_flag, n_val = [], 0, 0
    for s in steps:
        row = {l: near(l, s)[1] for l, _ in runs}
        have = {l: v for l, v in row.items() if v is not None}
        best = min(have, key=have.get) if have else None
        if best:
            lead_runs.append(best)
        line = f"{s:>8}"
        for l, _ in runs:
            v = row[l]
            if v is None:
                line += f"{'-':>11} "
                continue
            n_val += 1
            hot = args.flag and v > args.flag
            n_flag += bool(hot)
            line += f"{v:>11.5g}{'*' if hot else ' '}"
        print(line + (f"   {best}" if len(runs) > 1 and best else ""))

    if args.flag:
        n = n_flag
        print(f"\n  * above {args.flag:g}: {n} of {n_val} "
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
    p.add_argument("--list", nargs="?", const="*", default=None,
                   metavar="SUBSTRING",
                   help="list the tags in the first run and stop. 491 are "
                        "logged here, most per variable and level, so guessing "
                        "a name costs a round trip: --list grad_2.0 or "
                        "--list val_RMSE shows what is actually there")
    p.add_argument("--ratio", default=None, metavar="SUBSTRING",
                   help="compare every tag containing SUBSTRING between the "
                        "first two runs, as a mean over the tail of training, "
                        "sorted by ratio. 191 per-layer gradient norms are "
                        "logged and reading them one at a time is hopeless; "
                        "this answers 'are this run's gradients smaller "
                        "everywhere' in one line")
    p.add_argument("--tail", type=float, default=0.25,
                   help="fraction of the run --ratio averages over")
    p.add_argument("--tag", default="val_loss",
                   help="any logged tag: val_loss, train_loss_epoch, "
                        "gradient_2norm, or a per-layer grad norm")
    p.add_argument("--head", type=int, default=0,
                   help="only the first N shared steps - for watching the "
                        "start of a run")
    p.add_argument("--every", type=int, default=1,
                   help="print every Nth shared step; the last is always kept")
    p.add_argument("--max-step", type=int, default=0)
    p.add_argument("--tol", type=int, default=600,
                   help="how far a run's nearest logged step may be from the "
                        "grid before it prints blank. One validation interval "
                        "here is about 454 steps, so 600 matches at most one "
                        "point either side and never invents a value")
    p.add_argument("--flag", type=float, default=0.0,
                   help="mark values above this with *. For gradient_2norm use "
                        "the trainer's gradient_clip_val, which is 2 here")
    main(p.parse_args())
