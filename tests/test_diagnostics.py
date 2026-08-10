"""tools/steering.py and tools/find_center.py.

Both tools rest on a physical claim that is easy to state and easy to get wrong,
so each is checked against a synthetic field where the answer is known by
construction, and with a negative control that fails if the mechanism under test
is removed.
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, "tools", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load("plot_forecast")
st = _load("steering")
fc = _load("find_center")

N = 81
RES = 0.25
LON0, LAT0 = 130.0, 20.0
DEG_KM = 111.32


def make_field(vortex_offset_deg=(0.0, 0.0), uniform=(0.0, 0.0), vmax=40.0):
    """A domain with one axisymmetric cyclone embedded in a uniform flow.

    The vortex is cyclonic and exactly axisymmetric about its own centre, which
    is what lets the areal mean recover the uniform flow. Offsetting it is what
    lets the locators be checked against a known answer.
    """
    c = (N - 1) / 2.0
    jj, ii = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing='xy')
    lon2d = LON0 + jj * RES
    lat2d = LAT0 - ii * RES                      # latitude descends down the rows

    dx = (lon2d - (LON0 + vortex_offset_deg[0])) * np.cos(np.deg2rad(LAT0))
    dy = lat2d - (LAT0 + vortex_offset_deg[1])
    r = np.hypot(dx, dy)
    rm = 0.5                                     # radius of maximum wind, degrees
    vt = vmax * (r / rm) * np.exp(0.5 * (1.0 - (r / rm) ** 2))

    safe = np.where(r < 1e-9, 1e-9, r)
    u = uniform[0] - vt * dy / safe              # cyclonic in the northern hemisphere
    v = uniform[1] + vt * dx / safe

    sfc = np.zeros((N, N, len(pf.SFC)))
    sfc[..., pf.SFC.index('lon')] = lon2d
    sfc[..., pf.SFC.index('lat')] = lat2d
    sfc[..., pf.SFC.index('u10')] = 0.8 * u
    sfc[..., pf.SFC.index('v10')] = 0.8 * v
    sfc[..., pf.SFC.index('msl')] = 101_000.0 - 5_000.0 * np.exp(-(r / 1.5) ** 2)

    up = np.zeros((len(pf.LEVELS), N, N, len(pf.UPPER)))
    for k in range(len(pf.LEVELS)):
        up[k, :, :, pf.UPPER.index('u')] = u
        up[k, :, :, pf.UPPER.index('v')] = v
    return pf.Field(up, sfc)


# --------------------------------------------------------------------------
# steering.py - the areal mean must return the environment, not the vortex
# --------------------------------------------------------------------------

def test_areal_mean_recovers_the_flow_the_vortex_sits_in():
    """The whole diagnostic rests on the vortex cancelling over a centred disc.

    A 40 m/s cyclone on a 6 m/s background: if the cancellation did not hold the
    answer would be dominated by the vortex, and every steering number the tool
    prints would be meaningless.
    """
    f = make_field(uniform=(-6.0, 2.0))
    mask = st.disc(f, st.STEER_RADIUS_KM)
    u, v = st.areal_wind(f, 500, mask)
    assert u == pytest.approx(-6.0, abs=0.3)
    assert v == pytest.approx(2.0, abs=0.3)


def test_vortex_strength_does_not_change_the_areal_mean():
    """The mechanism itself: a stronger storm must not read as stronger steering.

    Same environment, no vortex against a 60 m/s one. If these differed, the tool
    would be reporting intensity as if it were steering, and the whole comparison
    against the storm's motion would be circular.
    """
    mask = st.disc(make_field(), st.STEER_RADIUS_KM)
    calm = st.areal_wind(make_field(uniform=(-6.0, 2.0), vmax=0.0), 500, mask)
    fierce = st.areal_wind(make_field(uniform=(-6.0, 2.0), vmax=60.0), 500, mask)
    assert calm == pytest.approx(fierce, abs=1e-3)


def test_cancellation_survives_drift_but_not_the_disc_edge():
    """Negative control, and a documented limit of the tool.

    The cancellation is exact while the circulation lies wholly inside the disc,
    which is why a storm drifting a degree or two from the array centre - the
    situation find_center.py exists to detect - does not corrupt the steering
    estimate. Push the vortex out towards the 4.5 degree edge and part of its
    circulation falls outside, the cancellation fails, and the number stops
    meaning anything. Read the steering output together with the measured drift.
    """
    mask_at = lambda f: st.disc(f, st.STEER_RADIUS_KM)
    near = make_field(vortex_offset_deg=(2.0, 0.0), uniform=(-6.0, 2.0))
    edge = make_field(vortex_offset_deg=(4.0, 0.0), uniform=(-6.0, 2.0))
    u_n, v_n = st.areal_wind(near, 500, mask_at(near))
    u_e, v_e = st.areal_wind(edge, 500, mask_at(edge))
    assert np.hypot(u_n + 6.0, v_n - 2.0) < 0.05
    assert np.hypot(u_e + 6.0, v_e - 2.0) > 0.3


def test_disc_radius_matches_the_requested_distance():
    f = make_field()
    mask = st.disc(f, st.STEER_RADIUS_KM)
    c = (N - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing='ij')
    r_km = np.hypot(xx, yy) * RES * DEG_KM
    assert r_km[mask].max() <= st.STEER_RADIUS_KM + 1e-6
    assert r_km[~mask].min() > st.STEER_RADIUS_KM


def test_translation_converts_a_known_track_to_metres_per_second():
    """A storm moving due west at a round rate must come back as that rate."""
    hours = list(range(0, 25, 6))
    speed_deg_per_h = 0.25                       # 1 degree every 4 hours
    centres = {h: (LON0 - speed_deg_per_h * h, LAT0) for h in hours}
    motion = st.translation(centres, hours)
    expect = -speed_deg_per_h * DEG_KM * np.cos(np.deg2rad(LAT0)) * 1000.0 / 3600.0
    for h in hours:
        assert motion[h][0] == pytest.approx(expect, rel=1e-6)
        assert motion[h][1] == pytest.approx(0.0, abs=1e-9)


def test_deep_layer_is_the_850_to_300_band():
    assert st.deep_layer(pf.LEVELS) == [850, 700, 600, 500, 400, 300]


# --------------------------------------------------------------------------
# find_center.py - locate the vortex, not the frame
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", fc.METHODS)
@pytest.mark.parametrize("offset", [(0.0, 0.0), (1.0, -0.75)])
def test_locate_finds_a_planted_vortex(method, offset):
    f = make_field(vortex_offset_deg=offset)
    lon, lat = fc.locate(f, method, search_deg=5.0)
    assert lon == pytest.approx(LON0 + offset[0], abs=0.2)
    assert lat == pytest.approx(LAT0 + offset[1], abs=0.2)


def test_the_wind_maximum_is_not_the_centre():
    """Why 'wind' is a minimum here and not a maximum, as upstream has it.

    On a vortex that is axisymmetric by construction, so the centre is not a
    matter of opinion, argmax of the 10 m wind lands in the eyewall tens of km
    away. The coupling repository's finding_LLAT_TC_center.py uses exactly that.
    """
    f = make_field()
    ws = np.hypot(f.s('u10'), f.s('v10'))
    i, j = np.unravel_index(np.nanargmax(ws), ws.shape)
    off = np.hypot((float(f.lon[i, j]) - LON0) * np.cos(np.deg2rad(LAT0)),
                   float(f.lat[i, j]) - LAT0) * DEG_KM
    assert off > 40.0

    lon, lat = fc.locate(f, 'wind_min', search_deg=5.0)
    assert np.hypot((lon - LON0) * np.cos(np.deg2rad(LAT0)),
                    lat - LAT0) * DEG_KM < 10.0


def test_subgrid_refinement_beats_the_nearest_cell():
    """Negative control for the parabolic fit.

    The vortex is planted half a cell off a grid point, where whole-cell argmax
    is worst. The drift being measured is tens of km and a cell is 28, so without
    this the tool would report quantisation as signal.
    """
    off = (0.5 * RES, 0.0)
    f = make_field(vortex_offset_deg=off)
    lon, _ = fc.locate(f, 'mslp', search_deg=5.0)

    z = np.where(np.isfinite(f.s('msl')), f.s('msl'), np.inf)
    _, j = np.unravel_index(np.argmin(z), z.shape)
    nearest_lon = float(f.lon[0, j])

    truth = LON0 + off[0]
    assert abs(lon - truth) < abs(nearest_lon - truth)
    assert abs(lon - truth) < 0.4 * RES


def test_declared_centre_is_the_mean_of_the_coordinate_channels():
    """The identity the whole Lagrangian frame rests on, over the polar disc.

    NaN outside the disc is deliberate; what remains is still symmetric about the
    centre, so the mean still lands on it. Filling those corners with anything
    else is what dragged an earlier forecast 3300 km west.
    """
    f = make_field()
    r = np.hypot((f.lon - LON0) * np.cos(np.deg2rad(LAT0)), f.lat - LAT0)
    f.lon[r > 10.0] = np.nan
    f.lat[r > 10.0] = np.nan
    lon, lat = fc.declared(f)
    assert lon == pytest.approx(LON0, abs=1e-6)
    assert lat == pytest.approx(LAT0, abs=1e-6)


def test_search_radius_excludes_a_stronger_system_further_out():
    """A trough in the outer domain must not be mistaken for the storm."""
    f = make_field()
    msl = f.sfc[..., pf.SFC.index('msl')]
    r = np.hypot((f.lon - LON0) * np.cos(np.deg2rad(LAT0)), f.lat - LAT0)
    msl[r > 7.0] -= 9_000.0                      # deeper than the TC, far away
    lon, lat = fc.locate(f, 'mslp', search_deg=5.0)
    assert lon == pytest.approx(LON0, abs=0.2)
    assert lat == pytest.approx(LAT0, abs=0.2)
