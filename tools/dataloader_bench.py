"""How many samples a second does the dataloader deliver, with no GPU at all?

The GPUs sampled at roughly 30 % utilisation with 8.5 % of their memory used,
which is what waiting for the CPU looks like: twelve cores feeding eight H200s,
with the polar resampling done per sample inside the worker. grid_sample is a
GPU kernel and it is being run on the CPU, at 38 ms a call and two calls per
sample at R = 80.

None of that needs a GPU to measure, and measuring it without one matters when
the allocation budget is spent. This iterates the real DataLoader - the real
dataset, the real transforms, the real worker count - and reports samples per
second. Run it on a login node or a CPU-only allocation.

What the number means. Eight H200s at the model's own speed want roughly
`batch_size x steps_per_second x 8` samples a second. If the dataloader delivers
less than that, every GPU-side change - more devices, a bigger model - buys
nothing, because the bottleneck does not move.

Use it to compare, not to predict: run it before a change to datasets.py and
after, on the same node with the same --workers, and the ratio is the honest
answer. Absolute numbers depend on how busy the login node is.

Usage
-----
    cd $HOME/LLAT_polar
    python tools/dataloader_bench.py --config config.yaml --overlay experiments/r80_420k.yaml

    # sweep the worker count to find where it stops helping
    for w in 2 4 8 10 12; do
        python tools/dataloader_bench.py --overlay experiments/r80_420k.yaml \
            --workers $w --batches 30
    done
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deep_merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def main(args):
    import yaml

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if args.overlay:
        with open(args.overlay, encoding='utf-8') as f:
            deep_merge(cfg, yaml.safe_load(f))
    data = dict(cfg.get('data') or {})

    # The config uses n_workers; the DataModule may take either name, so both
    # are set rather than guessing which one this version reads.
    if args.workers is not None:
        data['n_workers'] = args.workers
    if args.batch_size is not None:
        data['batch_size'] = args.batch_size

    print(f"config {args.config}"
          + (f" + {args.overlay}" if args.overlay else ""))
    print(f"  grid       {data.get('data_spatial_shape')}")
    print(f"  batch_size {data.get('batch_size')}   n_workers "
          f"{data.get('n_workers')}   persistent "
          f"{data.get('persistent_workers')}")

    # Imported late: utils.data_processor opens land.nc and needs metpy,
    # pysolar and xarray_regrid at import, so an early failure here should say
    # so rather than looking like a bug in this file.
    from utils.data_modules import ERA5TCDataModule

    dm = ERA5TCDataModule(**{k: v for k, v in data.items()
                             if k in _accepted(ERA5TCDataModule)})
    dm.setup('fit')
    dl = dm.train_dataloader()
    bs = data.get('batch_size', 1)

    print(f"\nwarming up {args.warmup} batches (worker start-up and the first "
          f"file reads are not throughput)...", flush=True)
    it = iter(dl)
    for _ in range(args.warmup):
        next(it)

    print(f"timing {args.batches} batches", flush=True)
    t0 = time.perf_counter()
    per = []
    for i in range(args.batches):
        t1 = time.perf_counter()
        next(it)
        per.append(time.perf_counter() - t1)
    total = time.perf_counter() - t0

    import numpy as np
    per = np.array(per)
    print(f"\n  {args.batches} batches of {bs} in {total:.1f} s")
    print(f"  {args.batches * bs / total:>8.1f} samples/s")
    print(f"  {args.batches / total:>8.2f} batches/s")
    print(f"  per batch: mean {1000*per.mean():.0f} ms, median "
          f"{1000*np.median(per):.0f}, p90 {1000*np.percentile(per,90):.0f}, "
          f"max {1000*per.max():.0f}")

    # A stalled worker pool shows as a long tail rather than a slow mean: most
    # batches come from the prefetch queue instantly and a few wait for a worker
    # to finish. Mean alone hides that.
    if per.max() > 5 * np.median(per):
        print("\n  The slowest batch is more than five times the median, which "
              "is a\n  starved prefetch queue rather than a uniformly slow "
              "transform. More\n  workers or a larger prefetch_factor is the "
              "lever, not a faster kernel.")
    print(f"\nEight GPUs at {args.ref_steps_per_s:g} steps/s would want "
          f"{8 * bs * args.ref_steps_per_s:.0f} samples/s.")
    print("Below that, adding devices or capacity changes nothing: the "
          "bottleneck does not move.")


def _accepted(cls):
    import inspect
    return set(inspect.signature(cls.__init__).parameters)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--overlay", default=None,
                   help="an experiments/*.yaml, merged over the config the same "
                        "way the job script does it")
    p.add_argument("--workers", type=int, default=None,
                   help="override n_workers; sweep this to find where it stops "
                        "helping")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5,
                   help="batches to discard: worker start-up and the first file "
                        "reads are not steady-state throughput")
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--ref-steps-per-s", type=float, default=4.0,
                   help="the model's own step rate, for the target line. "
                        "105,000 steps in 7.5 h is about 3.9")
    main(p.parse_args())
