"""Polar <-> Cartesian round trip (regression tests for B1 / B2 / B6).

Background
----------
The last two surface channels are `lon` / `lat`, and they are *predicted
variables*, not coordinate axes: the grid follows the storm (Lagrangian), so
"the TC moved" is implemented as "the lon/lat field shifted". Downstream code
therefore takes the MEAN of those fields as the TC centre, and uses it to decide
where in the FCNV2 global grid the LLAT field gets pasted.

B1 and B2 attack exactly that chain:
  B1  `polar_to_latlon(fill_value=0.0)` - the polar disc does not fill the 81x81
      square, so the four corners (1536 cells, 23.4%) were set to a physical 0.
  B2  `lonlat_uniformizer` locates the centre from the mean of the whole field,
      so those zeros diluted it by the area ratio: 130E was read as 99.6E, off by
      3,300 km, and the error compounds because the corrupted axis is written
      back into the state.

The fix is `fill_value=np.nan` plus nan-aware means. These tests pin down the
properties that break if either half is undone.

No onnx weights needed: this exercises the two transforms and lonlat_uniformizer
directly.
"""
import numpy as np
import pytest

from DLAMPty_inference import latlon_to_polar, polar_to_latlon
from utils.data_processor import lonlat_uniformizer

# Same values predict_one_step uses for the baseline model.
R, THETA, R_MAX = 201, 180, 40.0
N = 81
CENTER = (40.0, 40.0)
RES = 0.25
LON0, LAT0 = 130.0, 15.0


def make_lonlat():
    """A clean lon/lat field centred at (130E, 15N) with 0.25 deg spacing.

    Latitude DECREASES with row index, matching ERA5 and this project's data.
    The direction test below depends on that.
    """
    lon2d, lat2d = np.meshgrid(
        LON0 + (np.arange(N) - 40) * RES,
        LAT0 - (np.arange(N) - 40) * RES,
    )
    return np.stack([lon2d, lat2d], axis=-1)


def roundtrip(sfc, fill_value):
    polar, _, _ = latlon_to_polar(sfc, R=R, Theta=THETA, r_max=R_MAX)
    return polar_to_latlon(polar, output_shape=(N, N), r_max=R_MAX,
                           center_xy=CENTER, fill_value=fill_value)


# --------------------------------------------------------------------------
# Geometry: the disc does not fill the square. This is the root of B1.
# --------------------------------------------------------------------------

def test_circle_does_not_fill_square():
    """A radius-40 disc covers 76.6% of an 81x81 grid; the corners have no data.

    Not a bug, just geometry - but pinning it means nobody can make the question
    "what goes in the corners" disappear by quietly changing r_max.
    """
    yy, xx = np.meshgrid(np.arange(N) - 40.0, np.arange(N) - 40.0, indexing="ij")
    inside = np.hypot(xx, yy) <= R_MAX
    assert inside.sum() == 5025
    assert (~inside).sum() == 1536
    assert 0.76 < inside.mean() < 0.77
    # Corner-to-centre distance is sqrt(40^2+40^2) = 56.6 cells, well outside.
    assert np.hypot(40.0, 40.0) > R_MAX


# --------------------------------------------------------------------------
# B1: outside must be NaN, inside must survive
# --------------------------------------------------------------------------

def test_outside_is_nan_and_inside_is_exact():
    back = roundtrip(make_lonlat(), np.nan)
    yy, xx = np.meshgrid(np.arange(N) - 40.0, np.arange(N) - 40.0, indexing="ij")
    inside = np.hypot(xx, yy) <= R_MAX

    assert np.isnan(back[~inside]).all(), "everything outside the disc should be NaN"
    assert not np.isnan(back[inside]).any(), "nothing inside the disc should be NaN"

    truth = make_lonlat()
    # Bilinear interpolation, so not bit-identical, but far below one cell (0.25 deg).
    assert np.abs(back[inside] - truth[inside]).max() < 0.05


# --------------------------------------------------------------------------
# B2: the centre must survive, and the axes must not flip
# --------------------------------------------------------------------------

def test_center_survives_roundtrip():
    back = roundtrip(make_lonlat(), np.nan)
    lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)

    assert np.nanmean(lon) == pytest.approx(LON0, abs=0.01)
    assert np.nanmean(lat) == pytest.approx(LAT0, abs=0.01)


def test_axis_direction_preserved():
    """Longitude increasing, latitude decreasing.

    This looks trivial but is the easiest part of the nan-aware fix to miss.
    `lonlat_uniformizer` decides each axis's direction from the SIGN of
    `np.diff(...).mean()`. Leave those two means non-nan-aware and they evaluate
    to NaN; `specify_resolution * lon_res >= 0` is then always False and the
    longitude axis is silently reversed.
    """
    back = roundtrip(make_lonlat(), np.nan)
    lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)

    assert lon[1] - lon[0] == pytest.approx(+RES, abs=1e-9), "longitude must increase"
    assert lat[1] - lat[0] == pytest.approx(-RES, abs=1e-9), "latitude must decrease"


def test_center_stable_over_many_steps():
    """Autoregressive case: write the derived axis back and go round again.

    What made B2 dangerous is that it COMPOUNDS - with fill 0 the centre is
    multiplied by 0.766 per step, so 130E reaches 34E after five. Fixed, it must
    not drift at all.
    """
    cur = make_lonlat()
    for _ in range(5):
        back = roundtrip(cur, np.nan)
        lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)
        cur = np.stack(np.meshgrid(lon, lat), axis=-1)

    assert np.nanmean(cur[..., 0]) == pytest.approx(LON0, abs=0.01)
    assert np.nanmean(cur[..., 1]) == pytest.approx(LAT0, abs=0.01)


# --------------------------------------------------------------------------
# Negative control: prove these tests can actually see B1/B2
# --------------------------------------------------------------------------

def test_fill_zero_reproduces_the_bug():
    """Deliberately use the old fill_value=0.0 and reproduce the 0.766x collapse.

    Without this, the tests above cannot demonstrate they have any teeth.
    """
    back = roundtrip(make_lonlat(), 0.0)
    lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)

    ratio = 5025 / (N * N)                      # disc-to-square area ratio, ~0.766
    assert np.nanmean(lon) == pytest.approx(LON0 * ratio, rel=0.02)
    assert np.nanmean(lat) == pytest.approx(LAT0 * ratio, rel=0.02)
    # In distance terms: longitude is off by more than 3,000 km.
    err_km = (LON0 - np.nanmean(lon)) * 111 * np.cos(np.deg2rad(LAT0))
    assert err_km > 3000


# --------------------------------------------------------------------------
# Backward compatibility: the Cartesian path has no NaN and must be unchanged
# --------------------------------------------------------------------------

def test_cartesian_path_unchanged():
    """With no NaN present nanmean == mean, so Cartesian inference is unaffected."""
    sfc = make_lonlat()
    lon, lat = lonlat_uniformizer(sfc[:, :, 0], sfc[:, :, 1], True, RES)

    assert np.nanmean(lon) == pytest.approx(LON0, abs=1e-9)
    assert np.nanmean(lat) == pytest.approx(LAT0, abs=1e-9)
    assert lon[1] - lon[0] == pytest.approx(+RES, abs=1e-9)
    assert lat[1] - lat[0] == pytest.approx(-RES, abs=1e-9)
