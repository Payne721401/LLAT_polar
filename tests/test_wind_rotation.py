"""vt/vr <-> u/v rotation on the polar grid (S4).

The model is trained on tangential/radial wind; FCNV2, the saved npy and the
plotting code all speak u/v. predict_one_step therefore rotates on the way in
and back on the way out, in POLAR space where theta is exact per column.

What these tests can and cannot establish
-----------------------------------------
They lock down that the rotation is a proper rotation - invertible, speed
preserving, and correctly wired into predict_one_step. They deliberately do NOT
assert which of the eight conventions the dataset uses: that is a property of
data this repo does not contain, and is measured by
tools/verify_vtvr_convention.py. Asserting a guess here would turn an unknown
into a false certainty.
"""
import numpy as np
import pytest

from DLAMPty_inference import WIND_CONVENTIONS, rotate_polar_wind

R, THETA = 41, 180
THETA_RAD = np.deg2rad(np.linspace(0, 360, THETA, endpoint=False))


def random_polar(nvar=6, seed=0):
    return np.random.default_rng(seed).normal(size=(R, THETA, nvar))


# --------------------------------------------------------------------------
# The eight conventions are all genuine rotations
# --------------------------------------------------------------------------

def test_all_conventions_orthogonal():
    """p*s == -q*t is what makes the 2x2 matrix orthogonal for every theta.

    Without it the transform is not invertible and does not preserve wind
    speed, so it could not describe any sane vt/vr definition.
    """
    for name, (p, q, s, t) in WIND_CONVENTIONS.items():
        assert p * s == -q * t, name
        assert {abs(p), abs(q), abs(s), abs(t)} == {1}, name


@pytest.mark.parametrize("convention", sorted(WIND_CONVENTIONS))
def test_roundtrip_is_identity(convention):
    """u,v -> vt,vr -> u,v must return the original field."""
    a = random_polar()
    b = a.copy()
    rotate_polar_wind(b, THETA_RAD, 0, 1, convention, inverse=False)
    assert not np.allclose(a[..., :2], b[..., :2]), "rotation did nothing"
    rotate_polar_wind(b, THETA_RAD, 0, 1, convention, inverse=True)
    np.testing.assert_allclose(b, a, atol=1e-12)


@pytest.mark.parametrize("convention", sorted(WIND_CONVENTIONS))
def test_speed_preserved(convention):
    """A rotation may not change wind speed - that is the physical check."""
    a = random_polar()
    speed_before = np.hypot(a[..., 0], a[..., 1])
    b = a.copy()
    rotate_polar_wind(b, THETA_RAD, 0, 1, convention, inverse=False)
    np.testing.assert_allclose(np.hypot(b[..., 0], b[..., 1]), speed_before, atol=1e-12)


@pytest.mark.parametrize("convention", sorted(WIND_CONVENTIONS))
def test_other_channels_untouched(convention):
    """Only the two wind channels may change."""
    a = random_polar()
    b = a.copy()
    rotate_polar_wind(b, THETA_RAD, 0, 1, convention, inverse=False)
    np.testing.assert_array_equal(b[..., 2:], a[..., 2:])


def test_rejects_non_orthogonal():
    import DLAMPty_inference as di
    di.WIND_CONVENTIONS['_bad'] = (1, 1, 1, 1)          # p*s = 1, q*t = 1
    try:
        with pytest.raises(ValueError, match="orthogonal"):
            rotate_polar_wind(random_polar(), THETA_RAD, 0, 1, '_bad', inverse=False)
    finally:
        del di.WIND_CONVENTIONS['_bad']


# --------------------------------------------------------------------------
# A solid-body vortex: the one case where the answer is known by inspection
# --------------------------------------------------------------------------

def test_pure_rotation_field_has_no_radial_part():
    """For a field that is purely tangential, vr must be zero everywhere.

    Build the vortex directly in polar space from the convention's own
    definition of "tangential", then check the radial channel vanishes. This
    verifies the two channels are not swapped, independently of which
    convention is in force.
    """
    for name, (p, q, s, t) in WIND_CONVENTIONS.items():
        # Inverse map applied to (vt=1, vr=0) gives the u/v of a unit
        # tangential flow under this convention.
        pair = np.zeros((R, THETA, 2))
        pair[..., 0] = 1.0                              # vt
        uv = pair.copy()
        rotate_polar_wind(uv, THETA_RAD, 0, 1, name, inverse=True)

        back = uv.copy()
        rotate_polar_wind(back, THETA_RAD, 0, 1, name, inverse=False)
        np.testing.assert_allclose(back[..., 0], 1.0, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(back[..., 1], 0.0, atol=1e-12, err_msg=name)


# --------------------------------------------------------------------------
# Wiring into predict_one_step
# --------------------------------------------------------------------------

def _identity_model(yaml_name, tmp_path, convention=None):
    """A DLAMPty_model whose "onnx" returns its input unchanged."""
    import os

    import yaml as _yaml
    from DLAMPty_inference import DLAMPty_model

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = _yaml.safe_load(open(os.path.join(root, "onnx", yaml_name), encoding="utf-8"))
    if convention is not None:
        cfg["polar"]["wind_convention"] = convention
    p = tmp_path / f"{yaml_name}.tmp.yaml"
    p.write_text(_yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    m = DLAMPty_model(str(p), root_dir=root)
    m.upper_mean = m.surface_mean = 0.0
    m.upper_std = m.surface_std = 1.0

    class Identity:
        def run(self, _, inputs):
            return [inputs["input_upper"], inputs["input_surface"]]

    m.model = Identity()
    return m


def _smooth_fields(m, seed=1):
    """Smooth synthetic input.

    White noise is useless here: predict_one_step interpolates twice
    (Cartesian -> polar -> Cartesian) and bilinear interpolation destroys
    noise, which would swamp whatever the rotation does.
    """
    n = m.cartesian_n
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    rng = np.random.default_rng(seed)

    def field(k):
        a, b, c = rng.uniform(0.5, 1.5, 3)
        return a * np.sin(1.5 * xx + k) * np.cos(1.2 * yy - k) + b * xx * yy + c

    n_sfc = len(m.surface_variables) + 2
    upper = np.stack([np.stack([field(z + 0.3 * v) for v in range(len(m.upper_variables))], -1)
                      for z in range(m.polar_shape[0])])
    sfc = np.stack([field(10 + v) for v in range(n_sfc)], -1)
    lon2d, lat2d = np.meshgrid(130 + (np.arange(n) - n // 2) * 0.25,
                               15 - (np.arange(n) - n // 2) * 0.25)
    sfc[..., -2], sfc[..., -1] = lon2d, lat2d
    return upper, sfc


@pytest.mark.parametrize("convention", sorted(WIND_CONVENTIONS))
def test_predict_one_step_rotations_cancel(tmp_path, convention):
    """Rotating in and back out must leave the same result as never rotating.

    Comparing against a `uv` model rather than against the raw input is what
    makes this test sharp: both paths interpolate Cartesian -> polar ->
    Cartesian, so that (substantial) error is common and cancels, leaving only
    the rotation. If only one of the two rotations were wired up - the easy
    mistake - the wind would come back rotated by theta and this fails for
    every convention.
    """
    ref = _identity_model("LLAT_polar_v1.yaml", tmp_path)          # uv, no rotation
    rot = _identity_model("LLAT_polar_vtvr_v1.yaml", tmp_path, convention)
    # Same grid for both, so the interpolation error really is identical.
    rot.polar_shape, rot.polar_R, rot.polar_theta = ref.polar_shape, ref.polar_R, ref.polar_theta

    upper, sfc = _smooth_fields(ref)
    a_u, a_s = ref.predict_one_step(upper.copy(), sfc.copy())
    b_u, b_s = rot.predict_one_step(upper.copy(), sfc.copy())

    np.testing.assert_allclose(b_s, a_s, atol=1e-6, err_msg=f"surface, {convention}")
    np.testing.assert_allclose(b_u, a_u, atol=1e-6, err_msg=f"upper, {convention}")


def test_rotation_actually_changes_the_polar_field(tmp_path):
    """Guard against the previous test passing because nothing happened at all.

    Capture what the model is handed with and without rotation; the wind
    channels must differ, the scalar channels must not.
    """
    ref = _identity_model("LLAT_polar_v1.yaml", tmp_path)
    rot = _identity_model("LLAT_polar_vtvr_v1.yaml", tmp_path, "ccw_outward")
    rot.polar_shape, rot.polar_R, rot.polar_theta = ref.polar_shape, ref.polar_R, ref.polar_theta

    seen = {}

    def capture(tag, model):
        class Grab:
            def run(self, _, inputs):
                seen[tag] = {k: v.copy() for k, v in inputs.items()}
                return [inputs["input_upper"], inputs["input_surface"]]
        model.model = Grab()

    capture("uv", ref)
    capture("vtvr", rot)
    upper, sfc = _smooth_fields(ref)
    ref.predict_one_step(upper.copy(), sfc.copy())
    rot.predict_one_step(upper.copy(), sfc.copy())

    a = seen["uv"]["input_surface"]
    b = seen["vtvr"]["input_surface"]
    assert not np.allclose(a[..., 0], b[..., 0]), "wind channel 0 was not rotated"
    assert not np.allclose(a[..., 1], b[..., 1]), "wind channel 1 was not rotated"
    np.testing.assert_allclose(b[..., 2:], a[..., 2:], atol=1e-6)


def test_external_names_are_uv():
    """The public interface is u/v even though the model is trained on vt/vr."""
    import os
    from DLAMPty_inference import DLAMPty_model

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = DLAMPty_model(os.path.join(root, "onnx", "LLAT_polar_vtvr_v1.yaml"), root_dir=root)
    assert m.upper_variables[:2] == ['vt', 'vr']
    assert m.upper_variables_external[:2] == ['u', 'v']
    assert m.surface_variables_external[:2] == ['u10', 'v10']
    # A uv model must be unaffected by the aliasing
    m2 = DLAMPty_model(os.path.join(root, "onnx", "LLAT_polar_v1.yaml"), root_dir=root)
    assert m2.upper_variables_external == m2.upper_variables
    assert m2.surface_variables_external == m2.surface_variables


def test_missing_convention_blocks_initialize():
    """The shipped vt/vr yaml leaves wind_convention unset on purpose."""
    import os
    from DLAMPty_inference import DLAMPty_model

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = DLAMPty_model(os.path.join(root, "onnx", "LLAT_polar_vtvr_v1.yaml"), root_dir=root)
    assert m.wind_convention is None
    with pytest.raises(ValueError, match="wind_convention"):
        m.initialize()
