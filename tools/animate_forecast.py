"""Turn a forecast into an animation, with every run side by side in each frame.

A still frame at one lead answers "what does it look like there". The questions
actually being asked are about evolution - when does a model start to recurve,
at what lead does an artefact appear, does a feature grow or get advected - and
those are visible in a loop and nearly invisible in a contact sheet.

Frames come from plot_forecast.py, run once per lead as a subprocess. That is
slower than importing it and driving the figure directly, and it is deliberate:
the panel layout, the colour limits, the derived-variable handling and the ERA5
column all stay in one place, so a change there shows up here without this file
knowing anything about it. Rendering a 120 h forecast takes a couple of minutes,
which is cheaper than two implementations of the same figure disagreeing.

Output is a GIF by default because it needs no codec and opens anywhere. --mp4
is smaller and smoother if ffmpeg is on the path.

Usage
-----
    B=$HOME/LLAT_polar_runs; P=$HOME/LLAT_polar_runs_p1; T=$HOME/LLAT_polar_runs_t360
    E=/wk2/yungyun/FCNV2_TC
    python tools/animate_forecast.py \\
        --run "baseline=$B/202419W/one_way_couple_model_LLAT_polar_vtvr_v1/start_from_2024100700" \\
        --run "P1=$P/202419W/one_way_couple_model_LLAT_polar_p1_v1/start_from_2024100700" \\
        --run "t360=$T/202419W/one_way_couple_model_LLAT_polar_t360_v1/start_from_2024100700" \\
        --era5 $E/202419W/ERA5/for_DLAMPty --tc-id 202419W --init 2024100700

The GIF lands in analysis/figures/forecasts/<TCID>/<init>/fields.gif unless --out
says otherwise. Frames are kept in a `frames/` subdirectory beside it, so a rerun
skips what it already drew; pass --refresh to redraw them.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

_here = os.path.dirname(os.path.abspath(__file__))


def leads_available(run_dir):
    """Forecast hours present, read from the filenames rather than assumed."""
    out = []
    d = os.path.join(os.path.expanduser(run_dir), 'DLAMPty', 'forecast')
    for f in glob.glob(os.path.join(d, 'output_sfc_*h.npy')):
        m = re.search(r'output_sfc_(\d+)h\.npy$', f)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def render(args, leads, frame_dir):
    """One plot_forecast figure per lead, skipping frames already drawn."""
    os.makedirs(frame_dir, exist_ok=True)
    made = []
    for i, h in enumerate(leads):
        png = os.path.join(frame_dir, f"frame_{h:04d}h.png")
        made.append(png)
        if os.path.exists(png) and not args.refresh:
            continue
        cmd = [sys.executable, os.path.join(_here, 'plot_forecast.py'),
               '--tc-id', args.tc_id, '--init', args.init,
               '--lead', str(h), '--out', png]
        for r in args.run:
            cmd += ['--run', r]
        if args.era5:
            cmd += ['--era5', args.era5]
        # Held fixed across frames on purpose: plot_forecast picks these per
        # call, and a mask radius or dpi that drifts between leads makes the
        # animation flicker in a way that reads as a change in the forecast.
        cmd += ['--mask-radius', str(args.mask_radius), '--dpi', str(args.dpi)]
        if args.panels:
            cmd += ['--panels'] + [str(x) for x in args.panels]
        print(f"  [{i + 1}/{len(leads)}] +{h:03d} h", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(png):
            # Print the child's own message: a missing lead in one run, or a
            # variable the ERA5 file does not carry, is far more useful than
            # "subprocess failed".
            sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:] + "\n")
            raise SystemExit(f"plot_forecast failed at +{h} h; see above")
    return made


def write_gif(frames, out, fps, loop=0):
    from PIL import Image

    # Colour limits are per-frame in plot_forecast, so frames can differ in size
    # by a pixel or two when a colourbar label changes width. GIF requires them
    # identical, so everything is padded to the largest.
    imgs = [Image.open(f).convert('RGB') for f in frames]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    fixed = []
    for im in imgs:
        if (im.width, im.height) != (w, h):
            canvas = Image.new('RGB', (w, h), 'white')
            canvas.paste(im, (0, 0))
            im = canvas
        fixed.append(im.convert('P', palette=Image.ADAPTIVE, colors=256))
    fixed[0].save(out, save_all=True, append_images=fixed[1:],
                  duration=int(1000 / max(fps, 1)), loop=loop, optimize=True)


def write_mp4(frames, out, fps):
    listing = out + ".txt"
    with open(listing, 'w', encoding='utf-8') as fh:
        for f in frames:
            fh.write(f"file '{os.path.abspath(f)}'\nduration {1.0 / fps:.4f}\n")
        fh.write(f"file '{os.path.abspath(frames[-1])}'\n")
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', listing,
           '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(listing)
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed:\n" + p.stderr[-2000:] +
                         "\nDrop --mp4 to write a GIF instead, which needs no "
                         "codec.")


def main(args):
    runs = []
    for spec in args.run:
        label, _, path = spec.partition('=')
        path = os.path.expanduser(path or spec)
        if not os.path.isdir(path):
            raise SystemExit(
                f"no such directory: {path}\n"
                f"An empty or stray path means a shell variable in --run was "
                f"not set in this shell.")
        runs.append(path)

    # Only leads every run has: a frame missing one panel would make the
    # animation jump, and comparing runs at different times is the one thing
    # this figure exists to avoid.
    common = set(leads_available(runs[0]))
    for r in runs[1:]:
        common &= set(leads_available(r))
    leads = sorted(h for h in common
                   if h >= args.min_lead
                   and (args.max_lead is None or h <= args.max_lead)
                   and h % args.every == 0)
    if not leads:
        raise SystemExit("no forecast hours common to every --run")
    dropped = {os.path.basename(r): len(leads_available(r)) for r in runs}
    if len(set(dropped.values())) > 1:
        print(f"  runs have different lead counts {dropped}; using the "
              f"{len(leads)} in common")

    out = os.path.expanduser(args.out or os.path.join(
        "analysis", "figures", "forecasts", args.tc_id, args.init,
        "fields.mp4" if args.mp4 else "fields.gif"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    frame_dir = args.frames or os.path.join(os.path.dirname(out), "frames")

    print(f"{len(leads)} frames, +{leads[0]} to +{leads[-1]} h, "
          f"every {args.every * (leads[1] - leads[0]) // args.every if len(leads) > 1 else 0} h")
    frames = render(args, leads, frame_dir)

    if args.mp4:
        write_mp4(frames, out, args.fps)
    else:
        write_gif(frames, out, args.fps)
    size = os.path.getsize(out) / 1e6
    print(f"\nwrote {out}  ({size:.1f} MB)")
    print(f"frames kept in {frame_dir} - rerunning reuses them, --refresh redraws")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="label=path, repeatable; every run becomes a column")
    p.add_argument("--era5", help="TC-centred ERA5 directory, drawn as truth")
    p.add_argument("--tc-id", required=True)
    p.add_argument("--init", required=True, help="YYYYMMDDHH")
    p.add_argument("--out", default=None,
                   help="defaults to analysis/figures/forecasts/<TCID>/<init>/"
                        "fields.gif")
    p.add_argument("--frames", default=None,
                   help="where to keep the per-lead PNGs; defaults to frames/ "
                        "beside the output")
    p.add_argument("--every", type=int, default=6,
                   help="use every Nth forecast hour. The step is 3 h, so 6 is "
                        "two-hourly frames and 12 is half-daily")
    p.add_argument("--min-lead", type=int, default=0)
    p.add_argument("--max-lead", type=int, default=None)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--mask-radius", type=float, default=0.0,
                   help="passed to plot_forecast; held fixed across frames so "
                        "the animation does not flicker")
    p.add_argument("--panels", type=int, nargs="*", default=None,
                   help="which plot_forecast panels to draw; fewer panels make "
                        "a smaller file and a clearer loop")
    p.add_argument("--dpi", type=int, default=110,
                   help="lower than plot_forecast's 150: a 20-frame GIF at 150 "
                        "is tens of megabytes")
    p.add_argument("--mp4", action="store_true",
                   help="write mp4 via ffmpeg instead of a GIF: smaller and "
                        "smoother, but needs ffmpeg on the path")
    p.add_argument("--refresh", action="store_true",
                   help="redraw frames that already exist")
    main(p.parse_args())
