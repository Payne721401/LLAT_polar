"""Where a chained run has got to, and where it is aiming.

`job_scripts/train_gb200.sh` submits the next segment before it starts training,
and needs two numbers first: the step the newest checkpoint reached, and the
max_steps the run is aiming at. If the first is wrong the chain either stops
early - silently truncating the run - or never stops, spending its hops on jobs
that have nothing to do. Neither failure announces itself in the log.

Both numbers used to be scraped inline in the job script, where nothing could
test them. They live here so `tests/test_chain_resume.py` can, and so the
regexes have one home rather than two.

    python tools/chain_state.py --rundir runs/NAME --config config.yaml \
        --overlay experiments/NAME.yaml

prints "<step> <target>" on one line, which the shell reads with `read`.

The step comes from the checkpoint FILENAME rather than its contents. train.py
formats it as "...-e{epoch}-s{step}", and `save_last='link'` makes last.ckpt a
symlink to the newest real file, so following the link and reading the name
costs nothing where torch.load would move 305 MB. The cost of that choice is
that the two must stay in step: if train.py's `filename` ever loses its -s
suffix this returns None, which the caller must treat as "unknown" rather than
as zero.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import yaml

STEP_RE = re.compile(r"-s(\d+)$")


def checkpoint_step(path: str | None) -> int | None:
    """The training step a checkpoint holds, from its filename.

    Returns None when the path is missing or the name carries no step, which is
    a different answer from 0 and must not be collapsed into one: a fresh run is
    at zero, an unreadable name is unknown.
    """
    if not path or not os.path.exists(path):
        return None
    name = os.path.basename(os.path.realpath(path))
    if name.endswith(".ckpt"):
        name = name[: -len(".ckpt")]
    m = STEP_RE.search(name)
    return int(m.group(1)) if m else None


def newest_checkpoint(rundir: str) -> str | None:
    """The last.ckpt of the most recently written version_* under rundir.

    A resumed run gets a new version directory each segment, so "newest by
    mtime" and not "highest version number" - the numbers are job ids and do not
    sort numerically once they wrap or once a run is resumed out of order.
    """
    hits = glob.glob(os.path.join(
        rundir, "lightning_logs", "version_*", "checkpoints", "last.ckpt"))
    if not hits:
        return None
    return max(hits, key=lambda p: os.path.getmtime(p))


def target_steps(paths: list[str]) -> int | None:
    """trainer.max_steps after applying the configs in order, later winning.

    LightningCLI resolves several --config the same way, so this has to as well;
    reading only the overlay would miss a run that inherits max_steps from
    config.yaml, and reading only config.yaml would miss every run that does
    not.
    """
    target = None
    for p in paths:
        if not p:
            continue
        with open(p, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        value = (doc.get("trainer") or {}).get("max_steps")
        if value is not None:
            target = value
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--fresh", action="store_true",
                    help="report step 0 regardless of what is on disk, matching "
                         "FRESH=1 in the job script")
    args = ap.parse_args()

    step = 0 if args.fresh else checkpoint_step(newest_checkpoint(args.rundir))
    target = target_steps([args.config, *args.overlay])
    # -1 rather than an empty field: the shell reads this with `read STEP TARGET`
    # and an empty word would shift the second value into the first.
    print(f"{-1 if step is None else step} {-1 if target is None else target}")


if __name__ == "__main__":
    main()
