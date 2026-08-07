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


def test_coupling_root_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="coupling repo"):
        rcf.add_coupling_repo(str(tmp_path))


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
