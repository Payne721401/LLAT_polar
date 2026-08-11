"""run_coupled_forecast helpers.

The driver replaced a script with absolute paths and magic numbers baked in.
Parameterising it is only safe if the derived values reproduce the originals
exactly, so these tests compare against the literals that used to be there.
"""
import importlib.util
import os

import numpy as np
import pytest
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "run_coupled_forecast", os.path.join(ROOT, "run_coupled_forecast.py"))
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)


def _grid():
    return np.flip(np.linspace(-90, 90, 721)), np.linspace(0, 359.75, 1440)


def test_wp_indices_match_the_original_literals():
    """The saved sub-domain must be byte-identical to what the old script cut.

    It used np.argwhere(lat == 80) and friends, which relies on those exact
    floats existing in the grid. Deriving with argmin is robust to that but must
    land on the same indices, or previously saved output stops being comparable.
    """
    lat, lon = _grid()
    expected = (int(np.argwhere(lat == 80)[0][0]),
                int(np.argwhere(lat == -10)[0][0]),
                int(np.argwhere(lon == 80)[0][0]),
                int(np.argwhere(lon == 180)[0][0]))
    assert rcf.wp_indices() == expected == (40, 400, 320, 720)


def _ds(n):
    return xr.Dataset(coords=dict(latitude=np.arange(n, dtype=float),
                                  longitude=np.arange(n, dtype=float)))


def test_crop_reproduces_the_hardcoded_offset():
    """161 -> 81 must crop at offset 40, matching the old isel(40, 121)."""
    out = rcf.crop_to_domain(_ds(161), 81)
    assert out.sizes['latitude'] == out.sizes['longitude'] == 81
    assert float(out.latitude[0]) == 40.0


def test_crop_is_a_noop_when_already_right_sized():
    ds = _ds(81)
    assert rcf.crop_to_domain(ds, 81) is ds


def test_crop_rejects_too_small():
    with pytest.raises(ValueError, match="smaller"):
        rcf.crop_to_domain(_ds(80), 81)


def test_crop_rejects_uncentrable_margin():
    """An odd margin cannot be split evenly, so the TC would leave the centre.

    Silently rounding would break the whole Lagrangian premise - every later
    stage assumes the storm sits at the array centre.
    """
    with pytest.raises(ValueError, match="centre-crop"):
        rcf.crop_to_domain(_ds(162), 81)


def test_coupling_info_exposes_uv_names():
    """The exchange looks channels up by name and must see u10/v10, not vt10/vr10.

    predict_one_step presents u/v externally; passing the card's own names would
    raise ValueError inside the exchange for a vt/vr model.
    """
    from DLAMPty_inference import DLAMPty_model

    llat = DLAMPty_model(os.path.join(ROOT, "onnx", "LLAT_polar_vtvr_v1.yaml"),
                         root_dir=ROOT)
    info = rcf.coupling_info(llat)

    assert llat.surface_variables[:2] == ['vt10', 'vr10']
    assert info['surface_vars'][:2] == ['u10', 'v10']
    assert info['upper_vars'][:2] == ['u', 'v']
    # Every name the exchange looks up must resolve.
    for v in ('u10', 'v10', 't2m', 'msl', 'sp', 'tcwv'):
        assert v in info['surface_vars']
    for v in ('u', 'v', 'z', 't'):
        assert v in info['upper_vars']
    # Positions must be unchanged - only the labels differ.
    assert len(info['surface_vars']) == len(llat.surface_variables)
    assert info['surface_vars'][2:] == llat.surface_variables[2:]




def test_outside_mask_matches_the_coupling_radius():
    """The frozen ring must line up with the radius the exchange would replace.

    The exchange masks on dis_grid >= polar_bdy_mask_radius in DEGREES, with
    dis_grid = hypot(dx, dy) * 0.25. Standalone mode has to freeze exactly that
    region, or the two modes would not be comparable.
    """
    n, radius, res = 81, 8.0, 0.25
    mask = rcf.outside_mask(n, radius, res)

    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    expected = np.sqrt(((xx - 40) * res) ** 2 + ((yy - 40) * res) ** 2) >= radius
    np.testing.assert_array_equal(mask, expected)
    assert not mask[40, 40]                       # the centre is never frozen
    assert mask[0, 0] and mask[-1, -1]            # corners always are


def test_hold_boundary_restores_only_the_ring():
    """The interior keeps evolving; only the ring returns to the IC.

    Also why standalone does not emit NaN: polar_to_latlon leaves the corners
    NaN and, with no global model, this is the only thing that fills them in.
    The coordinate channels are exempt and are never NaN here - predict_one_step
    rewrites them as a complete uniform grid.
    """
    n = 81
    mask = rcf.outside_mask(n, 8.0, 0.25)
    up0 = np.zeros((13, n, n, 6))
    sfc0 = np.zeros((n, n, 20))
    up, sfc = np.ones_like(up0), np.ones_like(sfc0)
    # Corners come back NaN from the polar round trip, weather channels only.
    corner = np.hypot(*np.meshgrid(np.arange(n) - 40.0, np.arange(n) - 40.0,
                                   indexing="ij")[::-1]) > 40
    up[:, corner, :] = np.nan
    sfc[corner, :-2] = np.nan

    up, sfc = rcf.hold_boundary(up, sfc, up0, sfc0, mask)

    assert not np.isnan(up).any(), "the frozen ring should have cleared every NaN"
    assert not np.isnan(sfc).any()
    np.testing.assert_array_equal(up[:, mask, :], 0.0)      # ring reset to IC
    np.testing.assert_array_equal(up[:, ~mask, :], 1.0)     # interior untouched


def test_hold_boundary_leaves_the_coordinate_channels_alone():
    """lon/lat must keep evolving even where the weather is frozen.

    They are the moving frame, not weather: predict_one_step rewrites the whole
    field as a uniform grid on the predicted centre, and the next step recovers
    the centre by averaging it. Freezing part of it mixes two centres, and since
    both the frozen and unfrozen sets are centrally symmetric each contributes
    its own centre to the mean - so the storm advances by only the unfrozen
    fraction. At the coupling radius that is 49 %: half speed, silently.
    """
    n = 81
    mask = rcf.outside_mask(n, 8.0, 0.25)

    def meshgrid_at(lon0, lat0):
        return np.meshgrid(lon0 + (np.arange(n) - 40) * 0.25,
                           lat0 - (np.arange(n) - 40) * 0.25)

    sfc0 = np.zeros((n, n, 20))
    sfc0[..., -2], sfc0[..., -1] = meshgrid_at(130.0, 15.0)      # IC centre
    sfc = np.ones((n, n, 20))
    sfc[..., -2], sfc[..., -1] = meshgrid_at(132.0, 15.0)        # moved 2 deg east

    up = np.ones((13, n, n, 6))
    up, sfc = rcf.hold_boundary(up, sfc, np.zeros_like(up), sfc0, mask)

    # The centre must survive intact, not be dragged back toward the IC.
    assert np.mean(sfc[..., -2]) == pytest.approx(132.0, abs=1e-9)
    # Weather channels are still frozen.
    np.testing.assert_array_equal(sfc[mask, 0], 0.0)
    np.testing.assert_array_equal(sfc[~mask, 0], 1.0)


def test_freezing_lonlat_would_halve_the_motion():
    """Negative control: the defect this guards against, and its size.

    Without it the test above cannot show it measures anything.
    """
    n = 81
    mask = rcf.outside_mask(n, 8.0, 0.25)
    lon_ic = 130.0 + (np.arange(n) - 40) * 0.25
    lon_new = 132.0 + (np.arange(n) - 40) * 0.25
    field = np.tile(lon_new, (n, 1))
    field[mask] = np.tile(lon_ic, (n, 1))[mask]          # the old behaviour

    moved = np.mean(field) - 130.0
    assert 0.9 < moved < 1.1, moved                      # 2 deg asked, ~1 delivered
    assert mask.mean() == pytest.approx(0.512, abs=0.01)


def test_each_mode_writes_to_its_own_directory():
    """Two modes must not share an output path.

    They are not comparable runs, so a later one overwriting an earlier one is
    silent data loss - and the obvious first workflow is standalone followed by
    one-way on the same --out.

    The two-way name is fixed by the lab's plotting notebook; one_way_couple_model
    matches the sibling directory already used there.
    """
    import inspect

    src = inspect.getsource(rcf.main)
    assert "2_way_circle_couple_model_{version}" in src
    assert "one_way_couple_model_{version}" in src
    assert "standalone_{version}" in src

    names = {m: n for m, n in (
        ('two-way', '2_way_circle_couple_model'),
        ('one-way', 'one_way_couple_model'),
        ('standalone', 'standalone'))}
    assert len(set(names.values())) == 3


def test_corner_nan_reaches_the_disc_through_rim_interpolation():
    """NaN outside the disc is not harmless: the rim interpolates across it.

    polar_to_latlon leaves r > r_max undefined, which looks safe because
    latlon_to_polar only samples r <= r_max. But it samples bilinearly, so a
    point exactly on the rim averages four Cartesian cells and some of them lie
    just outside. Measured: 23.4 % NaN in the corners becomes 2.9 % of the polar
    array, in the outer two of 41 rings.

    That is enough to lose everything. Attention mixes every token, so one NaN
    makes the entire model output NaN - which is why one-way failed at the step
    after the first exchange while standalone, refilling every channel each step,
    ran to completion.
    """
    from DLAMPty_inference import latlon_to_polar

    n = 81
    yy, xx = np.meshgrid(np.arange(n) - 40.0, np.arange(n) - 40.0, indexing="ij")
    r = np.hypot(xx, yy)
    f = np.ones((n, n, 2))
    f[r > 40, 0] = np.nan

    p, _, _ = latlon_to_polar(f, R=41, Theta=180, r_max=40.0, center_xy=(40.0, 40.0))
    bad = np.isnan(p[..., 0])
    assert bad.any(), "the rim should pull NaN in; if not, the mechanism changed"
    rings = np.flatnonzero(bad.any(axis=1))
    assert rings.min() >= 38, rings          # outer rings only
    assert bad.mean() < 0.05                 # a small fraction, and still fatal


def test_fill_remaining_nan_makes_the_state_safe_to_step():
    """After patching, the polar conversion must be completely clean.

    The exchange covers six surface variables and five upper ones, and
    changing_additional_information recomputes several more, but d2m, tp,
    mtnlwrf, sst_filled and w are covered by neither.
    """
    from DLAMPty_inference import latlon_to_polar

    n = 81
    yy, xx = np.meshgrid(np.arange(n) - 40.0, np.arange(n) - 40.0, indexing="ij")
    r = np.hypot(xx, yy)

    up0, sfc0 = np.zeros((13, n, n, 6)), np.zeros((n, n, 20))
    up, sfc = np.ones((13, n, n, 6)), np.ones((n, n, 20))
    up[:, r > 40, :] = np.nan
    sfc[r > 40, :] = np.nan

    up, sfc = rcf.fill_remaining_nan(up, sfc, up0, sfc0)
    assert not np.isnan(up).any() and not np.isnan(sfc).any()
    # The interior must not have been touched.
    np.testing.assert_array_equal(sfc[r <= 40], 1.0)

    p, _, _ = latlon_to_polar(sfc, R=41, Theta=180, r_max=40.0, center_xy=(40.0, 40.0))
    assert not np.isnan(p).any(), "the rim still pulls NaN in"


def _ramps(n, lon0, lat0, res=0.25):
    """A uniform coordinate grid, as predict_one_step writes it."""
    return np.meshgrid(lon0 + (np.arange(n) - (n - 1) / 2) * res,
                       lat0 - (np.arange(n) - (n - 1) / 2) * res)


def test_fill_remaining_nan_keeps_one_centre_in_the_coordinate_channels():
    """The corners must be extended, not pasted from the initial condition.

    Weather outside the disc can be stale - it only matters through rim
    interpolation. lon/lat cannot: they are the frame, and mixing 23.4 % of the
    t=0 grid into 76.6 % of the current one leaves a field with two centres,
    whose mean is neither.
    """
    n = 81
    yy, xx = np.meshgrid(np.arange(n) - 40.0, np.arange(n) - 40.0, indexing="ij")
    outside = np.hypot(xx, yy) > 40.0

    sfc0 = np.zeros((n, n, 20))
    sfc0[..., -2], sfc0[..., -1] = _ramps(n, 130.0, 15.0)
    sfc = np.ones((n, n, 20))
    sfc[..., -2], sfc[..., -1] = _ramps(n, 132.0, 15.0)      # moved 2 deg east
    sfc[outside, :] = np.nan

    up = np.ones((13, n, n, 6))
    up[:, outside, :] = np.nan
    up, sfc = rcf.fill_remaining_nan(up, sfc, np.zeros_like(up), sfc0)

    assert not np.isnan(sfc).any()
    assert np.mean(sfc[..., -2]) == pytest.approx(132.0, abs=1e-9)
    assert np.mean(sfc[..., -1]) == pytest.approx(15.0, abs=1e-9)
    # Weather corners still come from the initial condition, as intended.
    np.testing.assert_array_equal(sfc[outside, 0], 0.0)


def test_pasting_the_ic_into_the_coordinate_corners_drags_the_frame_back():
    """Negative control, and the size of the defect it guards against.

    This is what fill_remaining_nan used to do. Unlike the hold_boundary version
    it does not stop the storm, it slows it to the disc's own area fraction -
    every step, and towards the origin rather than the previous position. A
    forecast that still moves does not look broken.
    """
    n = 81
    yy, xx = np.meshgrid(np.arange(n) - 40.0, np.arange(n) - 40.0, indexing="ij")
    outside = np.hypot(xx, yy) > 40.0

    lon_ic, _ = _ramps(n, 130.0, 15.0)
    lon_new, _ = _ramps(n, 132.0, 15.0)
    mixed = lon_new.copy()
    mixed[outside] = lon_ic[outside]                         # the old behaviour

    delivered = (np.mean(mixed) - 130.0) / 2.0
    assert delivered == pytest.approx(1.0 - outside.mean(), abs=0.01)
    assert 0.74 < delivered < 0.79, delivered                # 76.6 % of the step


def test_frame_speed_scale_multiplies_the_step_and_nothing_else():
    n = 81
    before = np.zeros((n, n, 20))
    before[..., -2], before[..., -1] = _ramps(n, 130.0, 15.0)
    after = np.full((n, n, 20), 7.0)
    after[..., -2], after[..., -1] = _ramps(n, 130.5, 15.2)   # the model's step

    scaled = rcf.rescale_frame_step(before, after.copy(), 1.45)
    assert np.mean(scaled[..., -2]) == pytest.approx(130.0 + 1.45 * 0.5, abs=1e-9)
    assert np.mean(scaled[..., -1]) == pytest.approx(15.0 + 1.45 * 0.2, abs=1e-9)
    # Still a uniform grid, just shifted: the spacing is untouched.
    assert scaled[0, 1, -2] - scaled[0, 0, -2] == pytest.approx(0.25, abs=1e-9)
    np.testing.assert_array_equal(scaled[..., :-2], 7.0)


def test_frame_speed_scale_of_one_changes_nothing():
    """The default must be exactly inert, or every unscaled run shifts."""
    n = 81
    before = np.zeros((n, n, 20))
    before[..., -2], before[..., -1] = _ramps(n, 130.0, 15.0)
    after = np.zeros((n, n, 20))
    after[..., -2], after[..., -1] = _ramps(n, 130.5, 15.2)
    np.testing.assert_allclose(rcf.rescale_frame_step(before, after.copy(), 1.0),
                               after, atol=0, rtol=0)
