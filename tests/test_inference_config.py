"""Polar grid parameterisation (S2) and the vt/vr model card (S3).

Design intent
-------------
The `polar:` block of onnx/*.yaml holds ONLY the three values that actually
exist in the training config.yaml:

    data_spatial_shape [Z, R, Theta]
    r_degree_max          (degrees)
    original_resolution   (degrees per cell)

Everything else - radius in cells, Cartesian domain size, sampling centre, the
spacing handed to lonlat_uniformizer - is derived. Switching grids is then a
three-line yaml edit, and a half-done change (new R, stale r_max) cannot be
expressed. That was the state before S2, when R/Theta/r_max were literals inside
predict_one_step.

No .onnx weights needed: these construct the object without calling initialize().
"""
import os

import numpy as np
import pytest
import yaml

from DLAMPty_inference import DLAMPty_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_UV = os.path.join(ROOT, "onnx", "LLAT_polar_v1.yaml")
YAML_VTVR = os.path.join(ROOT, "onnx", "LLAT_polar_vtvr_v1.yaml")


def write_yaml(tmp_path, **polar_overrides):
    """Copy the vt/vr yaml with an edited polar block into a temp file."""
    cfg = yaml.safe_load(open(YAML_VTVR, encoding="utf-8"))
    cfg["polar"].update(polar_overrides)
    p = tmp_path / "tmp.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------
# S3: contents of the vt/vr model card
# --------------------------------------------------------------------------

def test_vtvr_yaml_matches_training_config():
    """Variable lists and grid must agree with the training config.yaml.

    This is the quietest failure mode in the whole inference chain: get the
    variable ORDER wrong by one and the model reads humidity as temperature,
    with nothing to indicate anything is amiss.
    """
    cfg = yaml.safe_load(open(YAML_VTVR, encoding="utf-8"))
    train = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))

    assert cfg["upper_vars"] == train["data"]["upper_variables"]
    assert cfg["surface_vars"] == train["data"]["surface_variables"]
    assert cfg["polar"]["data_spatial_shape"] == train["data"]["data_spatial_shape"]
    assert cfg["polar"]["r_degree_max"] == train["data"]["r_degree_max"]
    assert cfg["polar"]["original_resolution"] == train["data"]["original_resolution"]
    # Same statistics file, or normalisation is done against a different baseline.
    assert os.path.basename(cfg["stat_mean_file"]) == os.path.basename(train["data"]["stat_mean_file"])
    assert os.path.basename(cfg["stat_std_file"]) == os.path.basename(train["data"]["stat_std_file"])


def test_units_length_matches_variables():
    for path in (YAML_UV, YAML_VTVR):
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        assert len(cfg["upper_units"]) == len(cfg["upper_vars"]), path
        assert len(cfg["surface_units"]) == len(cfg["surface_vars"]), path


def test_stat_files_exist():
    for path in (YAML_UV, YAML_VTVR):
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        for key in ("stat_mean_file", "stat_std_file"):
            assert os.path.exists(os.path.join(ROOT, cfg[key])), f"{path}: {cfg[key]}"


# --------------------------------------------------------------------------
# S2: derivation
# --------------------------------------------------------------------------

def test_derived_grid_uv():
    """The baseline: R=201 at 0.05 deg radial spacing, still an 81x81 domain."""
    m = DLAMPty_model(YAML_UV, root_dir=ROOT)
    assert (m.polar_R, m.polar_theta) == (201, 180)
    assert m.r_max_px == 40.0            # 10 deg / 0.25 deg per cell
    assert m.cartesian_n == 81           # 10*2/0.25 + 1
    assert m.center_xy == (40.0, 40.0)
    assert m.specify_resolution == 0.25
    assert m.wind_representation == "uv"


def test_derived_grid_vtvr():
    """This project's model: R=41, radial spacing matched to the source grid."""
    m = DLAMPty_model(YAML_VTVR, root_dir=ROOT)
    assert (m.polar_R, m.polar_theta) == (41, 180)
    assert m.r_max_px == 40.0
    assert m.cartesian_n == 81
    assert m.center_xy == (40.0, 40.0)


def test_grid_follows_yaml_without_touching_code(tmp_path):
    """Edit the three yaml values and the derived grid follows - the point of S2.

    Uses the planned 0.1-degree high-resolution case on purpose. The training
    side used to write `r_max = r_degree_max * 4`, hardcoding 1/0.25, which would
    give 40 cells instead of 100 here. Inference divides by the resolution, so
    this passes; if anyone converts it back to a multiplication, it fails.
    """
    p = write_yaml(tmp_path, data_spatial_shape=[13, 100, 360],
                   r_degree_max=10, original_resolution=0.1,
                   wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)
    assert (m.polar_R, m.polar_theta) == (100, 360)
    assert m.r_max_px == 100.0           # 10 / 0.1, not 10*4
    assert m.cartesian_n == 201          # 10*2/0.1 + 1
    assert m.center_xy == (100.0, 100.0)
    assert m.specify_resolution == 0.1


def test_training_dataset_derives_the_same_radius():
    """The training side must agree, or the model is served a different geometry.

    ERA5TCDataset now derives r_max_px the same way. Checked here rather than in
    a training test because the value only matters when the two ends agree.
    """
    import inspect

    from utils import datasets
    src = inspect.getsource(datasets.ERA5TCDataset.__init__)
    assert "self.r_max_px = self.r_degree_max / self.original_resolution" in src
    assert "r_degree_max*4" not in inspect.getsource(datasets)


# --------------------------------------------------------------------------
# Guards: a wrong setting must fail loudly, never run quietly
# --------------------------------------------------------------------------

def test_missing_polar_block_raises(tmp_path):
    cfg = yaml.safe_load(open(YAML_VTVR, encoding="utf-8"))
    del cfg["polar"]
    p = tmp_path / "no_polar.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(KeyError, match="polar"):
        DLAMPty_model(str(p), root_dir=ROOT)


def test_non_integer_radius_raises(tmp_path):
    """The radius has to land on a whole number of cells.

    `cartesian_n = int(2*r_degree_max/original_resolution) + 1` truncates. Unless
    `2*r_degree_max/original_resolution` is an integer, the domain half-width and
    the polar radius disagree: the disc is no longer inscribed in the square, the
    sampling geometry differs from training, and yet the shapes stay valid and
    the model runs without complaint.

    Example: 10 deg with 0.3 deg spacing gives a 33.33-cell radius in a domain
    that is only 33 cells wide. (10/0.25 -> 40 and 7/0.25 -> 28 both pass.)
    """
    p = write_yaml(tmp_path, r_degree_max=10, original_resolution=0.3,
                   wind_representation="uv")
    with pytest.raises(ValueError, match="inscribed"):
        DLAMPty_model(p, root_dir=ROOT)


def test_other_valid_resolutions_accepted(tmp_path):
    """Counter-check that the test above is not vacuous: exact ratios must pass."""
    p = write_yaml(tmp_path, data_spatial_shape=[13, 41, 180],
                   r_degree_max=7, original_resolution=0.25,
                   wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)
    assert m.r_max_px == 28.0
    assert m.cartesian_n == 57
    assert m.center_xy == (28.0, 28.0)


# --------------------------------------------------------------------------
# Wiring: predict_one_step must actually use the derived values
# --------------------------------------------------------------------------

class _FakeSession:
    """Stand-in for an onnxruntime session: records shapes, returns zeros."""

    def __init__(self, z, r, theta, n_upper, n_sfc):
        self.shape = (z, r, theta, n_upper)
        self.sfc_shape = (r, theta, n_sfc)
        self.seen = None

    def run(self, _, inputs):
        self.seen = {k: v.shape for k, v in inputs.items()}
        return [np.zeros((1,) + self.shape, dtype=np.float32),
                np.zeros((1,) + self.sfc_shape, dtype=np.float32)]


def test_predict_one_step_uses_derived_grid(tmp_path):
    """Change R in the yaml and the shape reaching the model must change too.

    Asserting on `m.polar_R` alone is not enough: the code can derive the right
    value and still pass a literal R=201 to latlon_to_polar, and every
    attribute-level test stays green. Verified by experiment - reverting to
    hardcoded values fails only this test.
    """
    R_ODD, THETA_ODD = 41, 180
    p = write_yaml(tmp_path, data_spatial_shape=[13, R_ODD, THETA_ODD],
                   r_degree_max=10, original_resolution=0.25,
                   wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)

    n_sfc = len(m.surface_variables) + 2          # ingest_space_info appends lon/lat
    m.model = _FakeSession(13, R_ODD, THETA_ODD, len(m.upper_variables), n_sfc)
    m.upper_mean = m.surface_mean = 0.0           # normalisation is not the point here
    m.upper_std = m.surface_std = 1.0

    n = m.cartesian_n
    lon2d, lat2d = np.meshgrid(130 + (np.arange(n) - n // 2) * 0.25,
                               15 - (np.arange(n) - n // 2) * 0.25)
    upper = np.zeros((13, n, n, len(m.upper_variables)))
    sfc = np.zeros((n, n, n_sfc))
    sfc[..., -2], sfc[..., -1] = lon2d, lat2d

    out_u, out_s = m.predict_one_step(upper, sfc)

    assert m.model.seen["input_upper"] == (1, 13, R_ODD, THETA_ODD, len(m.upper_variables))
    assert m.model.seen["input_surface"] == (1, R_ODD, THETA_ODD, n_sfc)
    assert out_u.shape == (13, n, n, len(m.upper_variables))
    assert out_s.shape == (n, n, n_sfc)


def test_predict_one_step_rejects_wrong_input_size(tmp_path):
    """An IC cropped differently from training must be refused up front."""
    p = write_yaml(tmp_path, wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)
    bad = 61                                       # should be 81
    with pytest.raises(ValueError, match="Cartesian grid"):
        m.predict_one_step(np.zeros((13, bad, bad, len(m.upper_variables))),
                           np.zeros((bad, bad, len(m.surface_variables) + 2)))
