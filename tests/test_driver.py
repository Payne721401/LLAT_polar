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
