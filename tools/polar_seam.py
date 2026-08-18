"""Is the blocky texture at the outer disc made by the model or by the coupling?

A --dump-polar of 202421W showed the model's own (r, Theta) output smooth
everywhere below about r index 30 and visibly blocky above it, in blocks roughly
6 columns by 4 rows - which is exactly patch_theta by patch_r. Index 32 of 40 is
r = 8.0 degrees, which is exactly --boundary-radius, where FCNV2 overwrites the
field in one-way mode and hold_boundary freezes the initial condition in
standalone.

The coincidence is suggestive and not evidence. Two things could make it:

  the coupling   a field overwritten outside 8 degrees meets the model's own
                 field at a sharp radial join, the patch embed tokenises across
                 it, and the patch grid becomes visible
  the model      the outer disc is simply where it is least constrained, and the
                 blocks are what an undertrained transformer looks like there

They are separated by running the same case twice and changing only whether
FCNV2 is coupled. --mode standalone has no exchange at all.

The measurement, rather than looking at pictures. A patch-grid artefact repeats
with period patch_theta along theta, so it puts energy at exactly one azimuthal
wavenumber - Theta/patch_theta, which is 30 for Theta=180 with patch 6. Weather
does not know about that wavenumber. So take the FFT along theta at each radius
and follow the amplitude at the patch harmonic: a flat profile that jumps at the
boundary radius is the coupling, a profile already high before it is the model.

Usage
-----
    python run_coupled_forecast.py ... --mode standalone --dump-polar /tmp/solo/raw
    python run_coupled_forecast.py ... --mode one-way --fcnv2-weight W \\
                                       --dump-polar /tmp/ow/raw

    python tools/polar_seam.py \\
        --dump "standalone=/tmp/solo/raw" --dump "one-way=/tmp/ow/raw" \\
        --boundary-radius 8 --r-max 10
"""
import argparse
import glob
import os

import numpy as np

# Surface channel order of the polar model card, for --var by name.
SFC = ['vt10', 'vr10', 't2m', 'd2m', 'msl', 'sp', 'tcwv', 'tp', 'mtnlwrf',
       'sst_filled', 'f', 'solar', 'hgt', 'landmask',
       'diurnal_sin', 'diurnal_cos', 'doy_sin', 'doy_cos', 'lon', 'lat']


def load(dump_dir, step):
    """One saved (R, Theta, C) surface array, by step index or the last one."""
    fs = sorted(glob.glob(os.path.join(os.path.expanduser(dump_dir),
                                       'polar_sfc_*.npy')))
    if not fs:
        raise SystemExit(
            f"no polar_sfc_*.npy in {dump_dir}. Was --dump-polar passed, and "
            f"did the forecast get past its first step?")
    return np.load(fs[step if step is not None else -1]), len(fs)


def patch_energy(a, patch_theta):
    """Amplitude at the patch harmonic, per radius, as a fraction of the total.

    Normalised by the field's own azimuthal variance at that radius, so a
    profile is comparable between radii where the field is strong and weak, and
    between variables with different units. A clean field sits near zero; a
    visible patch grid is percent-level.
    """
    Theta = a.shape[1]
    if Theta % patch_theta:
        raise SystemExit(f"Theta={Theta} is not divisible by patch_theta="
                         f"{patch_theta}; the harmonic is not a whole number")
    k = Theta // patch_theta            # the wavenumber a patch grid lives at
    f = np.fft.rfft(a - a.mean(axis=1, keepdims=True), axis=1)
    power = np.abs(f) ** 2
    total = power.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, power[:, k] / total, np.nan), k


def main(args):
    dumps = []
    for spec in args.dump:
        label, _, path = spec.partition('=')
        dumps.append((label or os.path.basename(path.rstrip('/')), path or spec))

    vi = SFC.index(args.var) if args.var in SFC else int(args.var)
    profiles, fields = {}, {}
    R = None
    for label, path in dumps:
        a, n = load(path, args.step)
        R = a.shape[0]
        prof, k = patch_energy(a[..., vi], args.patch_theta)
        profiles[label] = prof
        fields[label] = a[..., vi]
        print(f"{label:<14} {n} step(s), shape {a.shape}, "
              f"patch harmonic k={k}")

    r_deg = (np.arange(R) / max(R - 1, 1)) * args.r_max
    ib = int(np.argmin(np.abs(r_deg - args.boundary_radius)))

    print(f"\nenergy at the patch harmonic, as a fraction of azimuthal variance")
    print(f"{'r [deg]':>9}" + "".join(f"{l:>14}" for l, _ in dumps))
    print("-" * (9 + 14 * len(dumps)))
    for i in range(0, R, max(1, R // 20)):
        mark = "  <- boundary" if abs(i - ib) < max(1, R // 40) else ""
        row = "".join(f"{100 * profiles[l][i]:>13.2f}%" for l, _ in dumps)
        print(f"{r_deg[i]:>9.2f}{row}{mark}")

    print()
    for label, _ in dumps:
        p = profiles[label]
        inner = float(np.nanmean(p[:ib]))
        outer = float(np.nanmean(p[ib:]))
        ratio = outer / inner if inner > 0 else float('inf')
        print(f"{label:<14} inside {100 * inner:6.2f}%   outside "
              f"{100 * outer:6.2f}%   ratio {ratio:5.1f}x")

    if len(dumps) >= 2:
        (la, _), (lb, _) = dumps[0], dumps[1]
        oa = float(np.nanmean(profiles[la][ib:]))
        ob = float(np.nanmean(profiles[lb][ib:]))
        print()
        if oa > 0 and ob / oa > 2.0:
            print(f"{lb} has {ob / oa:.1f}x the patch-scale energy of {la}")
            print(f"outside {args.boundary_radius:g} deg. The coupling is making")
            print("it, so it is an inference problem: no retraining involved.")
        elif ob > 0 and oa / ob > 2.0:
            print(f"{la} has more than {lb}, which is backwards for a coupling")
            print("artefact and worth checking the two runs really differ only")
            print("in --mode.")
        else:
            print(f"{la} and {lb} carry comparable patch-scale energy outside")
            print(f"{args.boundary_radius:g} deg, so the coupling is NOT making")
            print("it - the model produces this on its own and the fix is in")
            print("training or in the tokenisation, not in the exchange.")

    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n = len(dumps)
        fig, ax = plt.subplots(1, n + 1, figsize=(5.5 * (n + 1), 4.2))
        for a_, (label, _) in zip(ax[:n], dumps):
            im = a_.imshow(fields[label], aspect='auto', origin='lower')
            a_.axhline(ib, color='r', lw=1.0, ls='--')
            a_.set_title(f"{label} — {args.var}")
            a_.set_xlabel("theta index"), a_.set_ylabel("r index")
            fig.colorbar(im, ax=a_, fraction=0.046)
        for label, _ in dumps:
            ax[n].plot(100 * profiles[label], r_deg, '-o', ms=3, label=label)
        ax[n].axhline(args.boundary_radius, color='r', lw=1.0, ls='--',
                      label=f"boundary {args.boundary_radius:g} deg")
        ax[n].set_xlabel("patch-harmonic energy [%]")
        ax[n].set_ylabel("radius [deg]")
        ax[n].legend(fontsize=8), ax[n].grid(alpha=0.3)
        out = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        fig.tight_layout(), fig.savefig(out, dpi=140)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", action="append", required=True,
                   help="label=DIR from --dump-polar, repeatable; give the "
                        "standalone run first and the coupled one second")
    p.add_argument("--var", default="msl",
                   help="surface channel, by name or index")
    p.add_argument("--step", type=int, default=None,
                   help="which dumped step; the last one by default, because "
                        "an artefact that accumulates is clearest there")
    p.add_argument("--patch-theta", type=int, default=6)
    p.add_argument("--boundary-radius", type=float, default=8.0,
                   help="must match the run's --boundary-radius")
    p.add_argument("--r-max", type=float, default=10.0,
                   help="the model card's polar.r_degree_max")
    p.add_argument("--out", default="analysis/figures/transform/polar_seam.png")
    main(p.parse_args())
