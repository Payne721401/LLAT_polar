"""tools/radial_profile.py — the diagnostic must separate its two targets.

A boundary seam and an under-constrained outer domain look the same on a map
(wiggly isobars, patchy wind) but have different radial signatures: a step at one
radius versus a gradual rise toward the rim. If the tool cannot tell them apart
it is not worth running, so that discrimination is what these tests check.
"""
import importlib.util
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "radial_profile", os.path.join(ROOT, "tools", "radial_profile.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
pf = rp.pf

N, RES = 81, 0.25


def make(msl):
    sfc = np.zeros((N, N, len(pf.SFC)))
    sfc[..., pf.SFC.index('msl')] = msl
    lon2d, lat2d = np.meshgrid(130 + (np.arange(N) - 40) * RES,
                               15 - (np.arange(N) - 40) * RES)
    sfc[..., pf.SFC.index('lon')] = lon2d
    sfc[..., pf.SFC.index('lat')] = lat2d
    return pf.Field(np.zeros((13, N, N, 6)), sfc)


def radius():
    yy, xx = np.meshgrid(np.arange(N) - 40.0, np.arange(N) - 40.0, indexing='ij')
    return np.hypot(xx, yy) * RES


def std_at(field, r_query):
    mid, _, std = rp.profile(field, lambda f: f.s('msl') / 100.0)
    return float(std[np.argmin(abs(mid - r_query))])


def test_axisymmetric_field_has_no_azimuthal_spread():
    """The null case: an axisymmetric field must read as ~no asymmetry.

    Not exactly zero. The axisymmetric component is removed by interpolating a
    binned mean profile, which is piecewise linear, so curvature within a bin
    leaves a residual - 0.03 hPa on a 30 hPa depression, a tenth of a percent.

    The number this test exists to hold down is what it was *before* detrending:
    1.29 hPa at r = 2, where the radial gradient is steepest. That is the regime
    where a plain within-bin standard deviation cries wolf precisely at the
    eyewall, reporting the storm's own structure as an artefact.
    """
    depth = 30.0                                   # hPa, peak to background
    f = make(101000 - 3000 * np.exp(-(radius() / 2) ** 2))
    for q in (2, 5, 8, 9.5):
        assert std_at(f, q) < 0.005 * depth, (q, std_at(f, q))


def test_a_seam_shows_as_a_step_at_one_radius():
    r = radius()
    f = make(101000 - 3000 * np.exp(-(r / 2) ** 2) + np.where(r >= 9.0, 300.0, 0.0))
    assert std_at(f, 9.0) > 0.5          # the discontinuity bin
    assert std_at(f, 8.0) < 0.01         # clean on either side
    assert std_at(f, 9.6) < 0.01


def test_under_constraint_shows_as_a_rise():
    r = radius()
    noise = np.random.default_rng(0).normal(size=(N, N)) * 80 * np.clip((r - 6) / 4, 0, None)
    f = make(101000 - 3000 * np.exp(-(r / 2) ** 2) + noise)
    a, b, c = std_at(f, 7.0), std_at(f, 8.5), std_at(f, 9.5)
    assert a < b < c, (a, b, c)          # monotone, not a step
    assert c > 3 * a


def test_sparse_bins_are_dropped():
    """Bins with too few cells say nothing; including them would add fake spikes.

    Beyond the polar disc most of a bin is NaN, so the outermost bins are exactly
    where a spurious spike would appear.
    """
    f = make(np.where(radius() > 10, np.nan, 101000.0))
    mid, _, _ = rp.profile(f, lambda x: x.s('msl') / 100.0)
    assert mid.max() < 12.0
