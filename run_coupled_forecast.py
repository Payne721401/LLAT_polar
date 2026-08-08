"""Run a coupled FCNV2 + LLAT forecast.

FCNV2 supplies the lateral boundary for LLAT; in two-way mode LLAT's inner TC
structure is written back into the global field. FCNV2 steps 6 h, LLAT steps 3 h
twice, then the two exchange.

This replaces the copy of inference_2_way_test_polar.py that lived in the
coupling repo with four absolute paths baked into it, pointed at the baseline u/v
model, and a hardcoded 40:121 crop. Everything is now a flag, and the crop is
derived from the model card.

Example
-------
    python run_coupled_forecast.py \
        --tc-id 202421W \
        --model-yaml onnx/LLAT_polar_vtvr_v1.yaml \
        --coupling-root ~/couple_FCNV2_LLAT \
        --fcnv2-weight /wk2/yungyun/code_space/FCNV2_test/weight \
        --data-root /wk2/yungyun/FCNV2_TC \
        --track-csv /wk2/yungyun/ERA5_2024_for_TC/TC_list_JMA_v2 \
        --out /wk2/<you>/LLAT_polar_runs \
        --mode one-way --max-starts 1

Three modes:

    standalone  LLAT alone, boundary frozen at the initial condition. Needs
                nothing but this repository and the .onnx - no FCNV2 weights, no
                GPU, no second checkout. The forecast degrades from the edge
                inwards, but the whole polar chain runs, so this is the right
                first thing to try.
    one-way     FCNV2 supplies LLAT's boundary; the global model is untouched.
    two-way     as above, plus LLAT's inner TC structure is written back.

Start with `--mode standalone --max-starts 1 --hours 24`.

Output layout
-------------
Deliberately identical to the original, so the lab's existing plotting notebook
reads it by changing one version string:

    {out}/{TC_ID}/{mode}_{onnx_version}/start_from_{YYYYMMDDHH}/
        FCNV2/forecast/output_weather_{HHH}h.npy      (coupled modes only)
        DLAMPty/forecast/output_upper_{HHH}h.npy
        DLAMPty/forecast/output_sfc_{HHH}h.npy
        run_meta.yaml

The two-way directory keeps its original name, 2_way_circle_couple_model_*, so
the lab's plotting notebook reads it unchanged; one-way uses that notebook's
sibling name. Each mode gets its own directory, or a later run would overwrite an
earlier one that is not comparable to it.

Vendored third-party code
-------------------------
global_model/FCNV2/ and interaction_tools/ are copied from the lab's coupling
repository so that this one runs on its own. FCNV2 is NVIDIA's FourCastNet v2
under Apache 2.0, which permits redistribution provided the licence travels with
it; see global_model/FCNV2/LICENSE_FourCastNetv2.

The copy is deliberate rather than ideal: the exchange helper is shared with the
Cartesian workflow upstream, so a fix made there will not arrive here. It is
small and rarely touched, and the alternative - depending on an unpushed branch
of another repository - proved worse in practice.
"""
import argparse
import datetime
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from DLAMPty_inference import DLAMPty_model

# pysolar re-emits these for every grid point of every step, burying real output
# in thousands of identical lines. Neither affects the result: the leap-second
# table stops in 2023 (worth well under a second of solar position), and the
# timezone notice is about numpy datetimes being naive, which they are by
# construction here since everything is UTC.
warnings.filterwarnings("ignore", message=".*leap seconds after.*")
warnings.filterwarnings("ignore", message=".*no explicit representation of timezones.*")

# Sub-domain of the global grid that gets saved, as index bounds on the standard
# 721x1440 (0.25 deg) FCNV2 grid. Western North Pacific: 10S-80N, 80E-180E.
# Saving the whole globe every 6 h would be ~50x larger for no added value.
WP_BOX = dict(lat_min=-10.0, lat_max=80.0, lon_min=80.0, lon_max=180.0)


def wp_indices():
    """Index bounds of WP_BOX on the FCNV2 grid."""
    lat = np.flip(np.linspace(-90, 90, 721))     # descending, as FCNV2 stores it
    lon = np.linspace(0, 359.75, 1440)
    return (int(np.argmin(abs(lat - WP_BOX['lat_max']))),
            int(np.argmin(abs(lat - WP_BOX['lat_min']))),
            int(np.argmin(abs(lon - WP_BOX['lon_min']))),
            int(np.argmin(abs(lon - WP_BOX['lon_max']))))


def crop_to_domain(ds, n):
    """Centre-crop an IC dataset to n x n.

    The original hardcoded isel(40:121), i.e. 161 -> 81. Deriving it from the
    model card means the same driver works for any r_degree_max /
    original_resolution, and a source that is already the right size is left
    alone instead of being silently mis-cropped.
    """
    ny = ds.sizes['latitude']
    nx = ds.sizes['longitude']
    if ny == n and nx == n:
        return ds
    if ny < n or nx < n:
        raise ValueError(f"IC is {ny}x{nx}, smaller than the required {n}x{n}")
    oy, ox = (ny - n) // 2, (nx - n) // 2
    if (ny - n) % 2 or (nx - n) % 2:
        raise ValueError(
            f"cannot centre-crop {ny}x{nx} to {n}x{n}: the margin is not even, so "
            "the TC would no longer sit at the array centre")
    print(f"  cropping IC {ny}x{nx} -> {n}x{n} (offset {oy},{ox})")
    return ds.isel(latitude=np.arange(oy, oy + n), longitude=np.arange(ox, ox + n))


def outside_mask(n, radius_deg, resolution):
    """Cells at least radius_deg from the domain centre."""
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing='ij')
    return np.hypot(xx, yy) * resolution >= radius_deg


def hold_boundary(up, sfc, up0, sfc0, mask, n_coord=2):
    """Restore the outer ring from the initial condition.

    Standalone mode's substitute for a global model. LLAT is a limited-area
    model, so something has to say what happens at the edge; freezing it at the
    IC is the crudest option and the forecast degrades from the outside in, but
    it needs no FCNV2 at all.

    It also repairs the corners. polar_to_latlon leaves everything outside the
    disc as NaN, and with no coupling nothing else would fill them back in.

    The last `n_coord` surface channels - lon and lat - are deliberately left
    alone. They are not weather, they ARE the moving frame: predict_one_step
    rewrites the whole field as a uniform grid centred on the predicted
    position, and the next step recovers the TC centre by averaging it. Freezing
    part of that field mixes two different centres, and because both the frozen
    and unfrozen sets are centrally symmetric, each contributes its own centre to
    the mean - so the storm advances by only the unfrozen fraction of what the
    model asked for. At the default radius that is 49 %: a storm moving at half
    speed, with nothing to indicate it.
    """
    up[:, mask, :] = up0[:, mask, :]
    sfc[mask, :-n_coord] = sfc0[mask, :-n_coord]
    return up, sfc


def write_run_meta(path, llat, args, init):
    """Record what produced this run, next to the output it produced.

    Without it, anything reading the npy has to hardcode the channel order,
    which then silently shifts by one the day a model card changes - rainfall
    plotted as total column water, with nothing to indicate it. The names stored
    are the EXTERNAL ones, matching what is actually in the arrays.
    """
    import yaml as _yaml

    meta = dict(
        model_yaml=os.path.abspath(args.model_yaml),
        onnx_version=llat.model_setting['onnx_version'],
        mode=args.mode,
        tc_id=args.tc_id,
        init=init.strftime('%Y%m%d%H'),
        hours=args.hours,
        step_hours=3,
        upper_vars=list(llat.upper_variables_external),
        surface_vars=list(llat.surface_variables_external) + ['lon', 'lat'],
        pressure_levels=list(llat.pressure_levels),
        cartesian_n=llat.cartesian_n,
        resolution=llat.original_resolution,
        polar_shape=list(llat.polar_shape),
        r_degree_max=llat.r_degree_max,
        boundary_radius=args.boundary_radius,
    )
    with open(os.path.join(path, 'run_meta.yaml'), 'w', encoding='utf-8') as f:
        _yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)


def coupling_info(llat):
    """The variable lists the exchange helper needs.

    It looks up channels by name ('u10', 'v10', 'u', 'v', ...). The model card
    lists the model's own names, which for a vt/vr model are vt10/vr10 and would
    raise ValueError. predict_one_step already presents u/v externally, so hand
    the exchange the external names.
    """
    info = dict(llat.model_setting)
    info['upper_vars'] = llat.upper_variables_external
    info['surface_vars'] = llat.surface_variables_external
    return info


def main(args):
    standalone = args.mode == 'standalone'
    print(f"mode          : {args.mode}")

    if standalone:
        # FCNV2 is not imported at all here, so this mode needs neither its
        # weights nor torch_harmonics. It exercises the entire polar chain -
        # conversion, wind rotation, centre estimation, additional-variable
        # recomputation - which makes it the right first thing to run.
        fcnv2 = transfer_FCNV2_DLAMPty_with_radius = None
    else:
        from global_model.FCNV2.FCNV2_inference import FCNV2_model
        from interaction_tools.FCNV2_DLAMPty_interaction import (
            transfer_FCNV2_DLAMPty_with_radius)

    llat = DLAMPty_model(args.model_yaml, root_dir=os.path.dirname(
        os.path.abspath(__file__)), device=args.llat_device)
    llat.initialize()
    info = coupling_info(llat)

    if not standalone:
        fcnv2 = FCNV2_model(args.fcnv2_weight, device=args.fcnv2_device)
        fcnv2.initialize()

    version = llat.model_setting['onnx_version']
    lat_max_i, lat_min_i, lon_min_i, lon_max_i = wp_indices()
    # Standalone freezes its own, narrower ring. boundary_radius exists to match
    # what the exchange replaces (8 deg), but there FCNV2 supplies evolving
    # values, whereas freezing at the IC pins 51 % of the domain to t=0 and lets
    # the boundary dominate the forecast.
    hold_radius = args.hold_radius or args.boundary_radius
    edge = outside_mask(llat.cartesian_n, hold_radius, llat.original_resolution)
    if standalone:
        frozen = 100 * edge.mean()
        print(f"holding r >= {hold_radius:g} deg at the IC ({frozen:.0f}% of the "
              "domain); lon/lat are excluded so the frame can still move")

    track = pd.read_csv(os.path.join(args.track_csv, f"{args.tc_id}.csv"))
    # FCNV2 steps 6 h, so only 00/12 UTC initial times line up with the cycle.
    track = track[~track['Hour'].isin([6, 18])]
    track['datetime'] = pd.to_datetime(track[['Year', 'Month', 'Day', 'Hour']])

    starts = track['datetime'].tolist()
    if args.max_starts:
        starts = starts[args.start_index:args.start_index + args.max_starts]
    print(f"initial times : {len(starts)} ({starts[0]} .. {starts[-1]})")

    # The mode is part of the directory name, or a later run silently overwrites
    # an earlier one - the two are not comparable and nothing would say so. The
    # two-way name is kept verbatim because the lab's plotting notebook reads it,
    # and one_way_couple_model matches the sibling directory already in use there.
    run_root = os.path.join(args.out, args.tc_id, {
        'two-way': f"2_way_circle_couple_model_{version}",
        'one-way': f"one_way_couple_model_{version}",
        'standalone': f"standalone_{version}",
    }[args.mode])

    for initial_time in starts:
        stamp = initial_time.strftime('%Y%m%d%H')
        print(f"\n=== {args.tc_id} IC {stamp} ===")

        llat_ic = os.path.join(args.data_root, args.tc_id, 'ERA5', 'for_DLAMPty',
                               f"{args.tc_id}_{stamp}_combined.nc")
        needed = [llat_ic]
        if not standalone:
            fcnv2_ic = os.path.join(args.data_root, args.tc_id, 'ERA5',
                                    'for_FCNV2', f"analysis_{stamp}.npy")
            needed.append(fcnv2_ic)
        for p in needed:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        base = os.path.join(run_root, f"start_from_{stamp}")
        llat_dir = os.path.join(base, 'DLAMPty', 'forecast')
        os.makedirs(llat_dir, exist_ok=True)
        write_run_meta(base, llat, args, initial_time)
        if not standalone:
            fcnv2_dir = os.path.join(base, 'FCNV2', 'forecast')
            os.makedirs(fcnv2_dir, exist_ok=True)
            fcnv2_state = np.load(fcnv2_ic)
            np.save(os.path.join(fcnv2_dir, "output_weather_000h"),
                    fcnv2_state[:, lat_max_i:lat_min_i, lon_min_i:lon_max_i])

        with xr.open_dataset(llat_ic) as ds:
            ds = crop_to_domain(ds, llat.cartesian_n)
            up, sfc = llat.IC_from_xarray_to_npy(ds)
        # Kept for standalone's frozen boundary.
        up0, sfc0 = up.copy(), sfc.copy()

        np.save(os.path.join(llat_dir, "output_upper_000h"), up)
        np.save(os.path.join(llat_dir, "output_sfc_000h"), sfc)

        for i in range(1, args.hours // 6 + 1):
            if not standalone:
                fcnv2_state = fcnv2.predict_one_step(fcnv2_state)
                np.save(os.path.join(fcnv2_dir, f"output_weather_{i*6:0>3}h"),
                        fcnv2_state[:, lat_max_i:lat_min_i, lon_min_i:lon_max_i])

            # Two 3 h LLAT steps per 6 h global step.
            for half in (3, 6):
                hh = i * 6 - 6 + half
                target = initial_time + datetime.timedelta(hours=hh)
                up, sfc = llat.predict_one_step(up, sfc)
                np.save(os.path.join(llat_dir, f"output_upper_{hh:0>3}h"), up)
                np.save(os.path.join(llat_dir, f"output_sfc_{hh:0>3}h"), sfc)
                up, sfc = llat.changing_additional_information(up, sfc, target)
                if standalone:
                    # Every step, not every 6 h: with no global model there is
                    # nothing else to stop the NaN corners entering the next input.
                    up, sfc = hold_boundary(up, sfc, up0, sfc0, edge)

            if not standalone:
                new_fcnv2, up, sfc = transfer_FCNV2_DLAMPty_with_radius(
                    fcnv2_state, up, sfc, info,
                    radius=args.feedback_radius,
                    polar_bdy_mask=True,
                    polar_bdy_mask_radius=args.boundary_radius)
                # One-way is two-way minus the write-back: LLAT still takes its
                # lateral boundary from FCNV2, the global model is left untouched.
                if args.mode == 'two-way':
                    fcnv2_state = new_fcnv2

            if i % 4 == 0:
                print(f"  +{i*6:0>3} h  {target:%Y-%m-%d %H}Z")

    print(f"\ndone -> {run_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tc-id", required=True, help="e.g. 202421W")
    p.add_argument("--model-yaml", default="onnx/LLAT_polar_vtvr_v1.yaml")
    p.add_argument("--fcnv2-weight", default=None,
                   help="directory holding weights.tar, global_means.npy, "
                        "global_stds.npy; not needed for --mode standalone")
    p.add_argument("--data-root", required=True,
                   help="holds {TC_ID}/ERA5/for_FCNV2 and .../for_DLAMPty")
    p.add_argument("--track-csv", required=True, help="directory of {TC_ID}.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=int, default=240, help="forecast length (default 10 days)")
    p.add_argument("--mode", choices=["standalone", "one-way", "two-way"],
                   default="two-way",
                   help="standalone runs LLAT alone with the boundary frozen at "
                        "the IC: no FCNV2, no GPU, no second checkout")
    p.add_argument("--feedback-radius", type=float, default=7.5,
                   help="degrees; LLAT writes back inside this radius (two-way)")
    p.add_argument("--boundary-radius", type=float, default=8.0,
                   help="degrees; FCNV2 replaces LLAT outside this radius")
    p.add_argument("--hold-radius", type=float, default=9.0,
                   help="standalone only: freeze the IC beyond this radius in "
                        "degrees. Narrower than --boundary-radius on purpose - "
                        "a frozen ring does not evolve, so a wide one dominates "
                        "the forecast. 23.4%% is the floor, the corners outside "
                        "the polar disc, which have no model output at all")
    p.add_argument("--fcnv2-device", default="cuda")
    p.add_argument("--llat-device", default="cpu")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-starts", type=int, default=0,
                   help="0 = every initial time; use 1 for a smoke test")
    a = p.parse_args()
    if a.mode != 'standalone' and not a.fcnv2_weight:
        p.error(f"--mode {a.mode} needs --fcnv2-weight (or use --mode standalone)")
    main(a)
