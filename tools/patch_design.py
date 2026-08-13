"""Cost out a patch/window design before spending GPU hours on it.

P1 is that `patch_shape` and both window sizes were inherited from a 201-point
radial grid and never re-derived for R = 41. The consequences are arithmetic, so
they can be checked in a second instead of discovered in hour three of a training
run - which is when a shape error would otherwise surface.

Three things this answers.

**Will it run at all.** DownSample reshapes the token grid to 5-D and hands it to
GetPad2D. When the radial or azimuthal token count is odd the pad is non-zero,
and `F.pad` in replicate mode does not accept a 4-element padding on a 5-D input,
so it raises. Both counts must be even. That is B8, and it is why 41 works with
patch_r = 8 (ceil(41/8) = 6) and 40 does not (5).

**How much of the work is real.** Window attention pads the token grid up to a
whole number of windows. At R = 41 the coarse stage holds 3 radial tokens inside
an 8-wide window, so five sixths of that axis is padding that is computed and
then masked. The "real" fraction below is the product over the three axes.

**What it costs.** Two different numbers, and confusing them is a trap this file
exists partly to close.

The `cost` column is *attention* work: quadratic in window volume, linear in the
number of windows. By that measure the candidates look much cheaper than the
current setting. They are not, end to end, because a smaller patch produces more
tokens - R = 40 with patch_r = 4 gives 10 radial tokens against 6 - and
everything outside attention scales with the token count. The 1.67x more tokens
very nearly cancels the attention saving.

--time measures the whole forward and backward instead. On CPU the wide-window
candidate lands near 0.91x and the uniform one near 0.86x, against 0.74x and
0.47x by attention alone - so the saving is real but a third of what the FLOP
count promises. CPU ratios are not GPU ratios either; use
`job_scripts/calibrate.sh` on the real hardware before setting max_steps from a
throughput assumption.

The numbers are derived arithmetically and then checked against the model itself:
--verify instantiates it, counts the blocks the analytic model assumed, and runs
a forward pass on CPU.

Usage
-----
    python tools/patch_design.py                 # the candidate table
    python tools/patch_design.py --verify        # and prove they run
"""
import argparse
import math
import os

# (Z, R, Theta) of the data, and the surface field the patch embedding appends as
# one extra vertical token.
Z_LEVELS = 13

CANDIDATES = [
    # name,                   R,   patch,      window1,      window2
    ("current",              41, (2, 8, 6),  (2, 10, 15), (2, 8, 10)),
    ("R40 p4 uniform w6/15", 40, (2, 4, 6),  (2, 6, 15),  (2, 6, 15)),
    ("R40 p4 w5/15",         40, (2, 4, 6),  (2, 5, 15),  (2, 5, 15)),
    ("R40 p4 w10/15 + w5/15",40, (2, 4, 6),  (2, 10, 15), (2, 5, 15)),
    ("R48 p4 uniform",       48, (2, 4, 6),  (2, 6, 15),  (2, 6, 15)),
    ("R41 p4 (odd, must fail)", 41, (2, 4, 6), (2, 6, 15), (2, 6, 15)),
    ("CLAUDE.md target 40x96", 40, (2, 4, 4), (2, 6, 12),  (2, 6, 12)),
]


def tokens(z_levels, R, Theta, patch):
    """Token grid after the patch embedding.

    The vertical count carries a +1 the other axes do not: the surface field is
    embedded separately and concatenated as one more level, which is why the
    current grid is 8 deep and not 7.
    """
    pz, pr, pt = patch
    return (math.ceil(z_levels / pz) + 1,
            math.ceil(R / pr),
            math.ceil(Theta / pt))


def stage(tok, window):
    """Padded grid, window count and real fraction for one attention stage."""
    padded = tuple(math.ceil(t / w) * w for t, w in zip(tok, window))
    n_win = tuple(p // w for p, w in zip(padded, window))
    real = 1.0
    for t, p in zip(tok, padded):
        real *= t / p
    volume = window[0] * window[1] * window[2]
    cost = n_win[0] * n_win[1] * n_win[2] * volume * volume
    return padded, n_win, real, cost


def plan(R, Theta, patch, w1, w2, blocks_fine=4, blocks_coarse=12):
    """Everything decidable without touching the model."""
    tok = tokens(Z_LEVELS, R, Theta, patch)
    # DownSample halves the two lateral axes and needs both to be even first.
    legal = (tok[1] % 2 == 0) and (tok[2] % 2 == 0)
    coarse = (tok[0], tok[1] // 2, tok[2] // 2)

    fine = stage(tok, w1)
    crs = stage(coarse, w2)
    total = blocks_fine * fine[3] + blocks_coarse * crs[3]

    # Radial span of one patch, against the radius of maximum wind. This is the
    # part that decides whether an eyewall can be represented at all.
    dr = 10.0 / (R - 1) if R > 1 else float('nan')      # degrees per cell
    return dict(tokens=tok, coarse=coarse, legal=legal, fine=fine, crs=crs,
                total=total, n_tokens=tok[0] * tok[1] * tok[2],
                patch_deg=patch[1] * dr, dr=dr)


def verify(name, R, Theta, patch, w1, w2, surface_vars=20, upper_vars=6):
    """Instantiate the real model and push a tensor through it.

    The arithmetic above can be right about the shapes and still miss something
    the implementation does, so the claim "this configuration runs" is only worth
    making after it has run.
    """
    import torch
    from models.pangu_polar import PanguPolarModel

    m = PanguPolarModel((Z_LEVELS, R, Theta), upper_vars, surface_vars,
                        [2, 6], [6, 12], 192, patch, w1, w2)
    m.eval()
    n_blocks = {}
    for mod in m.modules():
        if type(mod).__name__ == 'EarthSpecificBlock':
            key = tuple(getattr(mod, 'input_shape', ('?',)))
            n_blocks[key] = n_blocks.get(key, 0) + 1
    with torch.no_grad():
        u, s = m(torch.randn(1, Z_LEVELS, R, Theta, upper_vars),
                 torch.randn(1, R, Theta, surface_vars))
    params = sum(p.numel() for p in m.parameters())
    return u.shape, s.shape, params, n_blocks


def main(args):
    # Both --time and --verify import the model, so the repository root goes on
    # the path once here rather than inside whichever branch happens to run.
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    base = None
    print(f"{'design':<24} {'tokens':>12} {'coarse':>10} {'real fine':>10} "
          f"{'real coarse':>12} {'cost':>7} {'patch':>8}")
    print("-" * 92)

    for name, R, patch, w1, w2 in CANDIDATES:
        Theta = 96 if patch[2] == 4 else 180
        p = plan(R, Theta, patch, w1, w2)
        if base is None:
            base = p['total']
        if not p['legal']:
            odd = 'r' if p['tokens'][1] % 2 else 'theta'
            print(f"{name:<24} {'x'.join(map(str, p['tokens'])):>12} "
                  f"{'':>10} {'':>10} {'':>12} {'':>7}   "
                  f"CRASHES: {odd} tokens = "
                  f"{p['tokens'][1] if odd == 'r' else p['tokens'][2]} is odd (B8)")
            continue
        print(f"{name:<24} {'x'.join(map(str, p['tokens'])):>12} "
              f"{'x'.join(map(str, p['coarse'])):>10} "
              f"{100*p['fine'][2]:>9.0f}% {100*p['crs'][2]:>11.0f}% "
              f"{p['total']/base:>6.2f}x {p['patch_deg']:>7.2f}°")

    print("\nwhat the columns mean")
    print("  tokens       the grid after patch embedding, Z x r x theta; Z carries")
    print("               a +1 because the surface field becomes one more level")
    print("  coarse       after DownSample halves the two lateral axes")
    print("  real         fraction of attention positions holding a real token")
    print("               rather than padding, at that stage")
    print("  cost         attention work relative to the current setting, as")
    print("               n_windows x window_volume^2 summed over blocks")
    print("  patch        radial degrees spanned by one patch. The radius of")
    print("               maximum wind is 0.3-0.5 deg: a patch wider than that")
    print("               cannot represent an eyewall, whatever the training")

    if args.time:
        import time
        import torch
        print("\nend-to-end forward + backward on CPU. The attention column "
              "above does not\npredict it: a smaller patch makes more tokens, "
              "and everything outside\nattention scales with them.")
        base_t = None
        for name, R, patch, w1, w2 in CANDIDATES:
            Theta = 96 if patch[2] == 4 else 180
            if not plan(R, Theta, patch, w1, w2)['legal']:
                continue
            from models.pangu_polar import PanguPolarModel
            m = PanguPolarModel((Z_LEVELS, R, Theta), 6, 20, [2, 6], [6, 12],
                                192, patch, w1, w2)
            u, sfc = torch.randn(1, Z_LEVELS, R, Theta, 6), torch.randn(1, R, Theta, 20)
            a, b = m(u, sfc); (a.sum() + b.sum()).backward(); m.zero_grad()
            t0 = time.time()
            for _ in range(3):
                a, b = m(u, sfc); (a.sum() + b.sum()).backward(); m.zero_grad()
            dt = (time.time() - t0) / 3
            base_t = base_t or dt
            print(f"  {name:<24} {dt:>6.2f} s   {dt/base_t:>5.2f}x")

    if args.verify:
        print("\nverifying on CPU — instantiate, count blocks, forward")
        for name, R, patch, w1, w2 in CANDIDATES:
            Theta = 96 if patch[2] == 4 else 180
            expect_ok = plan(R, Theta, patch, w1, w2)['legal']
            if not expect_ok:
                # Run it anyway. A predicted crash that is never executed is a
                # claim, not a check, and the whole table rests on this rule.
                try:
                    verify(name, R, Theta, patch, w1, w2)
                    print(f"  {name:<24} DID NOT CRASH — the B8 rule is wrong")
                except Exception as e:
                    print(f"  {name:<24} crashed as predicted: "
                          f"{type(e).__name__}: {str(e).splitlines()[0]}")
                continue
            try:
                us, ss, params, blocks = verify(name, R, Theta, patch, w1, w2)
                nb = sum(blocks.values())
                print(f"  {name:<24} ok   upper {tuple(us)}  surface {tuple(ss)}  "
                      f"{params/1e6:.1f} M params  {nb} blocks")
            except Exception as e:
                print(f"  {name:<24} FAILED  {type(e).__name__}: "
                      f"{str(e)[:80]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true",
                   help="also build each model and run a forward pass")
    p.add_argument("--time", action="store_true",
                   help="measure forward+backward wall-clock, which is the "
                        "number that matters and which the attention column "
                        "does not predict")
    main(p.parse_args())
