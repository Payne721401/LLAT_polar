"""How far apart are training and validation loss, and is that gap growing?

A validation loss on its own says which run is better. It does not say why, and
in particular it does not say whether a run has room left in it. The gap between
the training and validation curves does: a model that fits its training set much
better than its validation set is memorising, and giving it more steps or more
capacity makes that worse rather than better.

The number that motivated this: the R = 41 baseline finished at train 0.21697
against val 0.24997, a gap of 15.2 %, on 14,522 samples for 24.1 M parameters
over 231 epochs. That is the yardstick every later run is read against, and it is
why "just raise max_steps" is not automatically the right answer.

Reads the TensorBoard event files directly, with no tensorboard import. The
parser below is about sixty lines and works anywhere Python does, which matters
because this gets run on the cluster inside the training environment and on a
laptop against copied-back files, and those two do not have the same packages.

A run that was interrupted and resumed has more than one version directory. Pass
the run directory and they are all read and merged on step, which is the whole
reason this takes a run rather than a file.

Usage
-----
    # one run
    python tools/train_val_gap.py runs/p1_wide

    # the comparison that is actually wanted
    python tools/train_val_gap.py runs/prod_lr5e-5 runs/p1_wide

    # write the merged curves out for plotting elsewhere
    python tools/train_val_gap.py runs/p1_wide --csv analysis/p1_curves.csv
"""
import argparse
import glob
import os
import struct

# What Lightning logs them as. train_loss_epoch is the per-epoch reduction of
# the step-wise train_loss; comparing the step-wise one against val_loss would
# compare a single batch against the whole validation set.
TRAIN_KEYS = ("train_loss_epoch", "train_loss", "loss_epoch")
VAL_KEYS = ("val_loss", "val_loss_epoch")

# A jump this long between consecutive events is downtime, not training. Steps
# take under two minutes here, so an hour is far beyond any real stall and well
# under the shortest requeue seen.
IDLE_GAP_S = 3600.0


def _varint(buf, i):
    """One protobuf base-128 varint. Returns (value, next index)."""
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf):
    """Walk a protobuf message, yielding (field_number, wire_type, payload).

    Payload is the raw bytes for wire type 2, and the decoded integer for the
    fixed-width and varint types. Enough for what Event and Summary need and
    deliberately not more.
    """
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        elif wire == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i:i + ln], i + ln
        elif wire == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            return                      # group types; not in these files
        yield num, wire, val


def read_events(path):
    """Every (tag, step, value) scalar in one TFRecord-framed event file.

    Framing is: uint64 length, uint32 CRC of the length, the payload, uint32 CRC
    of the payload. The CRCs are masked CRC32C and are skipped - a corrupt file
    truncates mid-record and the length read fails, which is caught, and that is
    the only failure mode seen in practice (a job killed mid-write).
    """
    out = []
    with open(path, "rb") as fh:
        blob = fh.read()
    i, n = 0, len(blob)
    while i + 12 <= n:
        (length,) = struct.unpack_from("<Q", blob, i)
        i += 12                                     # length + its CRC
        if i + length + 4 > n:
            break                                   # truncated tail
        rec, i = blob[i:i + length], i + length + 4
        step = None
        wall = None
        summaries = []
        for num, _wire, val in _fields(rec):
            if num == 1:                            # Event.wall_time, a double
                wall = struct.unpack('<d', val)[0]
            elif num == 2:                          # Event.step
                step = val
            elif num == 5:                          # Event.summary
                summaries.append(val)
        if step is None:
            continue
        for summ in summaries:
            for num, _wire, val in _fields(summ):
                if num != 1:                        # Summary.value
                    continue
                tag = None
                simple = None
                for vn, _vw, vv in _fields(val):
                    if vn == 1:                     # Value.tag
                        tag = vv.decode("utf-8", "replace")
                    elif vn == 2:                   # Value.simple_value
                        simple = struct.unpack("<f", vv)[0]
                if tag is not None and simple is not None:
                    out.append((tag, step, simple, wall))
    return out


def curves(run_dir):
    """Merge every event file under a run into {tag: {step: value}}.

    Later files win on a repeated step, which is what a resume should do: the
    steps replayed after a restart are the authoritative ones.
    """
    pats = (os.path.join(run_dir, "**", "events.out.tfevents.*"),
            os.path.join(run_dir, "events.out.tfevents.*"))
    files = sorted({f for p in pats for f in glob.glob(p, recursive=True)},
                   key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            f"no events.out.tfevents.* under {run_dir}. Lightning writes them to "
            f"{run_dir}/lightning_logs/version_*/ - check the run directory, and "
            f"remember the logs are the ones the trainer wrote, not job_logs/.")
    merged = {}
    for f in files:
        for tag, step, val, _wall in read_events(f):
            merged.setdefault(tag, {})[step] = val
    return merged, files


def curves_timed(run_dir):
    """As curves(), plus the wall-clock second each point was written at.

    Every TFRecord event carries a wall_time, which is the only elapsed-time
    signal in the file: train.py logs elapsed_time_hours with logger=False, so
    it never reaches the event log at all. Times are made relative to the first
    event of the run, and a resume shows up as a jump - the gap between the two
    jobs is real elapsed time but not compute, so it is reported rather than
    silently included.
    """
    import glob as _glob

    pats = (os.path.join(run_dir, "**", "events.out.tfevents.*"),
            os.path.join(run_dir, "events.out.tfevents.*"))
    files = sorted({f for p in pats for f in _glob.glob(p, recursive=True)},
                   key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no events.out.tfevents.* under {run_dir}")
    merged, wall = {}, {}
    t0 = None
    for f in files:
        for tag, step, val, w in read_events(f):
            merged.setdefault(tag, {})[step] = val
            if w is not None:
                t0 = w if t0 is None else min(t0, w)
                wall[step] = w
    if t0 is None:
        return merged, wall, files, 0.0

    # Remove the queue. A resumed run has a gap in wall_time between the job
    # that died and the job that continued it, and that gap is elapsed time but
    # not compute: P1 read as 87.2 hours against t360's 6.7 because eighty of
    # them were spent waiting after a full disk killed job 257976. Plotting that
    # against validation loss draws an eighty-hour flat line and makes the
    # wall-clock panel useless for exactly the runs that needed rescuing.
    #
    # Any jump longer than `idle_gap` is treated as downtime and subtracted, so
    # the axis becomes compute time. The total removed is returned rather than
    # hidden, because a run that needed six hours of requeueing is worth knowing
    # about even though the number does not belong on the x-axis.
    order = sorted(wall)
    removed = 0.0
    prev = wall[order[0]]
    shift = 0.0
    fixed = {}
    for k in order:
        w = wall[k]
        if w - prev > IDLE_GAP_S:
            shift += w - prev
            removed += w - prev
        fixed[k] = (w - t0 - shift) / 3600.0
        prev = w
    return merged, fixed, files, removed / 3600.0


def pick(merged, keys):
    for k in keys:
        if merged.get(k):
            return k, merged[k]
    return None, {}


def report(run_dir, args):
    merged, files = curves(run_dir)
    tname, train = pick(merged, TRAIN_KEYS)
    vname, val = pick(merged, VAL_KEYS)

    print(f"\n{run_dir}")
    print(f"  {len(files)} event file(s), {len(merged)} tags")
    if not val:
        print(f"  no validation scalar found. tags: {sorted(merged)[:12]}")
        return None
    if not train:
        print(f"  no per-epoch training scalar found - only {sorted(merged)[:12]}.")
        print("  Without one the gap cannot be computed; the run still has a")
        print("  best val below.")

    best_step = min(val, key=val.get)
    print(f"  train tag {tname!r}, val tag {vname!r}")
    print(f"  best val {val[best_step]:.5f} at step {best_step}"
          f"  (last step logged: {max(val)})")

    if not train:
        return None

    # Training and validation are not logged on identical steps, so each
    # validation point is paired with the nearest training point at or before
    # it. Pairing forward would compare a validation loss against training the
    # model had not done yet.
    tsteps = sorted(train)

    def nearest(s):
        prev = [t for t in tsteps if t <= s]
        return prev[-1] if prev else tsteps[0]

    rows = []
    for s in sorted(val):
        t = nearest(s)
        gap = 100.0 * (val[s] - train[t]) / train[t]
        rows.append((s, train[t], val[s], gap))

    print(f"\n  {'step':>8}{'train':>10}{'val':>10}{'gap':>9}")
    print("  " + "-" * 37)
    # Head and tail, plus the best-val row, rather than every epoch.
    show = rows[:2] + rows[-args.tail:]
    seen = set()
    for s, tr, vl, gp in show + [r for r in rows if r[0] == best_step]:
        if s in seen:
            continue
        seen.add(s)
        mark = "  <- best val" if s == best_step else ""
        print(f"  {s:>8}{tr:>10.5f}{vl:>10.5f}{gp:>8.1f}%{mark}")

    final = [r for r in rows if r[0] == best_step][0]
    print(f"\n  gap at best val: {final[3]:.1f}%"
          f"   (baseline R=41 was 15.2%)")

    # Is the gap opening? Only the converged part of the run can answer that.
    #
    # A first version compared the first third against the last and called both
    # runs "OPENING", which was wrong and worth recording. Early in training the
    # gap is NEGATIVE - the baseline starts at -37 % - because train_loss_epoch
    # averages over an epoch during which the model was still improving, while
    # val_loss is measured once at the end of it. So the training number is
    # stale and looks worse than it is. That transient dominates any comparison
    # that includes it, and it says nothing about memorisation.
    #
    # So: drop the warmup, then split what remains in half.
    tail = [r for r in rows if r[0] >= args.converged * max(val)]
    if len(tail) >= 4:
        half = len(tail) // 2
        g_early = sum(r[3] for r in tail[:half]) / half
        g_late = sum(r[3] for r in tail[half:]) / (len(tail) - half)
        v_early = sum(r[2] for r in tail[:half]) / half
        v_late = sum(r[2] for r in tail[half:]) / (len(tail) - half)
        drift = 100.0 * (v_late - v_early) / v_early

        span = 100 * (1 - args.converged)
        print(f"\n  over the last {span:.0f}% of training")
        print(f"    gap        {g_early:>8.1f}% -> {g_late:>8.1f}%")
        print(f"    validation {v_early:>9.5f} -> {v_late:>9.5f}  ({drift:+.2f}%)")
        print("  ", end="")

        # VALIDATION DECIDES, not the gap. A second wrong verdict came from
        # reading the gap on its own: both production runs widened their gap
        # over the final half - the baseline 12.6 % to 14.9 % - and were called
        # overfitting, while validation was still FALLING 1 % over the same
        # span. A model fitting everything better, training faster than
        # validation, widens the gap and is learning, not memorising.
        #
        # Overfitting is validation getting worse. The gap only qualifies the
        # answer once validation has stopped improving.
        if drift < -0.1:
            print("STILL IMPROVING: validation is falling. Not overfitting,")
            print("  whatever the gap does - a widening gap here just means the")
            print("  training set improves faster, which is what fitting looks")
            print("  like. Stopping early would leave something on the table.")
        elif drift > 0.25:
            print("OVERFITTING: validation has turned upward. This is past the")
            print("  useful point; the best checkpoint is behind the last one.")
        elif g_late > g_early + 1.0:
            print("CONVERGED, and the gap is still widening: the run is buying")
            print("  training fit that validation does not see. More steps of")
            print("  the same would not pay.")
        else:
            print("CONVERGED: validation and the gap are both flat. Whether more")
            print("  steps would help is a question about the learning-rate")
            print("  schedule - a cosine that reached zero cannot use them -")
            print("  rather than about capacity.")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".",
                    exist_ok=True)
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("run,step,train,val,gap_pct\n")
            for s, tr, vl, gp in rows:
                fh.write(f"{run_dir},{s},{tr:.6f},{vl:.6f},{gp:.3f}\n")
        print(f"  wrote {args.csv}")
    return final


def main(args):
    finals = {}
    for run in args.runs:
        r = report(os.path.expanduser(run), args)
        if r:
            finals[run] = r
    if len(finals) > 1:
        print(f"\n{'run':<28}{'train':>10}{'val':>10}{'gap':>9}")
        print("-" * 57)
        for run, (_s, tr, vl, gp) in finals.items():
            print(f"{os.path.basename(run.rstrip('/')):<28}"
                  f"{tr:>10.5f}{vl:>10.5f}{gp:>8.1f}%")
        # Read the two runs against each other rather than leaving it to the
        # reader, because the interesting case is easy to walk past: a run whose
        # TRAINING loss is worse and whose validation loss is better has not
        # bought anything by fitting harder - it generalises better outright,
        # which is the strongest form the result can take.
        names = list(finals)
        (t0, v0), (t1, v1) = ((finals[n][1], finals[n][2]) for n in names[:2])
        a, b = (os.path.basename(n.rstrip('/')) for n in names[:2])
        print()
        if v1 < v0 and t1 > t0:
            print(f"{b} fits the training set WORSE than {a} ({t1:.5f} against")
            print(f"{t0:.5f}) and validates BETTER ({v1:.5f} against {v0:.5f}).")
            print("None of its gain came from memorising, so it is a property of")
            print("the architecture and should extend to data neither run saw.")
        elif v1 < v0 and (v0 - v1) > (t0 - t1):
            print(f"{b} improves validation more than training, so the gain is")
            print("mostly generalisation rather than fit.")
        elif v1 < v0:
            print(f"{b} improved both, and improved training by more. Some of the")
            print("gain is extra fit and may not extend; check the gap column.")
        else:
            print(f"{b} did not improve on {a}'s validation loss.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+",
                   help="run directories, e.g. runs/prod_lr5e-5 runs/p1_wide")
    p.add_argument("--tail", type=int, default=6,
                   help="how many of the final epochs to print")
    p.add_argument("--converged", type=float, default=0.9,
                   help="fraction of training to discard before judging the "
                        "trend. The early gap is negative and swamps any window "
                        "containing it, and even the last half still spans a lot "
                        "of ordinary learning; 0.9 looks at the plateau, which "
                        "is where the question is actually asked")
    p.add_argument("--csv", help="write the merged per-epoch curves here")
    main(p.parse_args())
