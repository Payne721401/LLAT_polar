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
        summaries = []
        for num, _wire, val in _fields(rec):
            if num == 2:                            # Event.step
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
                    out.append((tag, step, simple))
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
        for tag, step, val in read_events(f):
            merged.setdefault(tag, {})[step] = val
    return merged, files


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

    # Is the gap opening? Compare the first and last thirds of the run.
    third = max(1, len(rows) // 3)
    early = sum(r[3] for r in rows[:third]) / third
    late = sum(r[3] for r in rows[-third:]) / third
    print(f"  gap early {early:.1f}%  ->  late {late:.1f}%", end="  ")
    if late > early + 2.0:
        print("OPENING: the run is memorising, and more steps make it worse.")
    elif late < early - 2.0:
        print("closing: still generalising, more steps may pay.")
    else:
        print("flat: the gap is not the thing limiting this run.")

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
        print("\nA lower val with a LARGER gap bought its improvement by fitting")
        print("the training set harder, and will not extend. A lower val with the")
        print("same gap is a real gain and leaves room for more steps.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+",
                   help="run directories, e.g. runs/prod_lr5e-5 runs/p1_wide")
    p.add_argument("--tail", type=int, default=6,
                   help="how many of the final epochs to print")
    p.add_argument("--csv", help="write the merged per-epoch curves here")
    main(p.parse_args())
