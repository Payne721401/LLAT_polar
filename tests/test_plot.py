"""tools/plot_forecast.py helpers."""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "plot_forecast", os.path.join(ROOT, "tools", "plot_forecast.py"))
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

N = 81
RES = 0.25
LON0, LAT0 = 130.0, 15.0


def grid():
    return np.meshgrid(LON0 + (np.arange(N) - 40) * RES,
                       LAT0 - (np.arange(N) - 40) * RES)


def field_with_wind(u, v, level=850):
    sfc = np.zeros((N, N, len(pf.SFC)))
    lon2d, lat2d = grid()
    sfc[..., pf.SFC.index('lon')] = lon2d
    sfc[..., pf.SFC.index('lat')] = lat2d
    up = np.zeros((len(pf.LEVELS), N, N, len(pf.UPPER)))
    up[pf.LEVELS.index(level), :, :, pf.UPPER.index('u')] = u
    up[pf.LEVELS.index(level), :, :, pf.UPPER.index('v')] = v
    return pf.Field(up, sfc)


# --------------------------------------------------------------------------
# The channel lists must agree with the model card
# --------------------------------------------------------------------------

def test_channel_order_matches_the_model_card():
    """A silent mislabelling otherwise: every panel indexes by position.

    The plotting script hardcodes the surface order so it can read a bare npy,
    which means a change to the card would quietly shift every field by one -
    precipitation plotted as total column water, and nothing to indicate it.
    The card lists the model's own names, so vt10/vr10 map to u10/v10 here,
    matching what predict_one_step actually writes out.
    """
    import yaml

    card = yaml.safe_load(open(os.path.join(ROOT, "onnx", "LLAT_polar_vtvr_v1.yaml"),
                               encoding="utf-8"))
    alias = {'vt': 'u', 'vr': 'v', 'vt10': 'u10', 'vr10': 'v10'}
    expected_sfc = [alias.get(v, v) for v in card['surface_vars']] + ['lon', 'lat']
    expected_upper = [alias.get(v, v) for v in card['upper_vars']]

    assert pf.SFC == expected_sfc
    assert pf.UPPER == expected_upper
    assert pf.LEVELS == card['pressure_levels']


# --------------------------------------------------------------------------
# Vorticity
# --------------------------------------------------------------------------

def test_vorticity_exact_for_solid_body_rotation():
    """Solid-body rotation has vorticity 2*omega everywhere - the one case with
    an analytic answer, so it pins both the sign and the metric factors."""
    omega = 2e-5
    lon2d, lat2d = grid()
    m_per_deg = 111_320.0
    dx = (lon2d - LON0) * m_per_deg * np.cos(np.deg2rad(lat2d))
    dy = (lat2d - LAT0) * m_per_deg
    f = field_with_wind(-omega * dy, omega * dx)

    z = f.vorticity(850)[20:61, 20:61]
    np.testing.assert_allclose(z, 2 * omega, rtol=1e-6)


def test_vorticity_sign_is_cyclonic_positive():
    """Counter-clockwise flow must come out positive.

    Latitude descends with row index here, so a sign slip in the dy term would
    flip this while leaving the magnitude right - and on a Northern Hemisphere
    TC the result would look plausible until compared with anything else.
    """
    omega = 2e-5
    lon2d, lat2d = grid()
    m_per_deg = 111_320.0
    dx = (lon2d - LON0) * m_per_deg * np.cos(np.deg2rad(lat2d))
    dy = (lat2d - LAT0) * m_per_deg
    f = field_with_wind(-omega * dy, omega * dx)
    assert f.vorticity(850)[40, 40] > 0


def test_vorticity_zero_for_uniform_flow():
    f = field_with_wind(np.full((N, N), 12.0), np.full((N, N), -5.0))
    np.testing.assert_allclose(f.vorticity(850)[5:-5, 5:-5], 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------

def test_mask_outside_blanks_beyond_radius():
    f = field_with_wind(np.zeros((N, N)), np.zeros((N, N)))
    out = pf.mask_outside(f, np.ones((N, N)), 9.5)
    assert np.isnan(out[0, 0])                  # corner is 14.1 deg away
    assert not np.isnan(out[40, 40])            # centre survives
    # The radius is in degrees, so it must scale with the grid spacing rather
    # than being a cell count.
    assert np.isnan(out[40, 0]) == (40 * RES > 9.5)


def test_mask_outside_disabled_returns_input():
    f = field_with_wind(np.zeros((N, N)), np.zeros((N, N)))
    a = np.ones((N, N))
    assert pf.mask_outside(f, a, 0) is a


def test_panels_are_well_formed():
    for p in pf.PANELS:
        assert {'label', 'shade', 'cmap', 'unit', 'wind'} <= set(p), p['label']
        assert p['wind'][0] in ('stream', 'quiver'), p['label']
        # A symmetric colour scale and a zero anchor are contradictory.
        assert not (p.get('sym') and p.get('zero_based')), p['label']
