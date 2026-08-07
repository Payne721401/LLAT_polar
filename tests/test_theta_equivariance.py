"""PanguPolarModel: smoke, determinism, and a theta-roll equivariance diagnostic.

Run with (cwd = repo root):
    python -m pytest tests/ -v

Real tests (these pass):
    test_smoke        - the model runs forward, shapes are right, no NaN
    test_determinism  - the same input twice in eval mode gives identical output

Diagnostic (xfail; NOT a gate):
    test_theta_equivariance - equivariance under a theta roll. Marked xfail on
    purpose, for reasons established by experiment in 2026-07:

      1. Equivariance is not the acceptance criterion for the polar grid. What
         actually matters is that the theta seam is continuous, and this
         transformer - non-overlapping patch embed plus window attention - turns
         out to be seamless by construction. Circular vs zero padding, and
         splitting vs keeping the cross-seam mask, all measured the same.
      2. Swin window attention is equivariant only to rolls of a whole number of
         WINDOWS, not of patches. The granularity is k = patch_theta *
         window_theta, and it additionally needs an even window_theta plus a
         single window at the coarse level. The (15, 10) windows used here meet
         none of that, so no non-trivial k in one revolution is equivariant -
         only the full k=180.
      3. See CLAUDE.md, "test commands". This test is kept as a numerical
         diagnostic, not as a pass/fail gate.

    (If the grid or windows are ever changed to something equivariant, drop the
    xfail and set k to that design's real granularity.)
"""
import pytest
import torch

from models.pangu_polar import PanguPolarModel

# Matches config.yaml
CFG = dict(
    data_spatial_shape=(13, 201, 180),
    upper_vars=6,
    surface_vars=20,
    depths=[2, 6],
    heads=[6, 12],
    embed_dim=192,
    patch_shape=(2, 8, 6),
    window_size1=(2, 10, 15),
    window_size2=(2, 8, 10),
)
AX_U, AX_S = 3, 2   # the Theta axis of (B,Z,R,Theta,C) and (B,R,Theta,C)
TOL = 1e-5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = PanguPolarModel(**CFG)
    m.eval()   # disable DropPath
    return m


@pytest.fixture(scope="module")
def inputs():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(1, 13, 201, 180, 6, generator=g)
    s = torch.randn(1, 201, 180, 20, generator=g)
    return u, s


@pytest.fixture(scope="module")
def base_out(model, inputs):
    with torch.no_grad():
        return model(*inputs)


def test_smoke(base_out, inputs):
    out_u, out_s = base_out
    assert out_u.shape == inputs[0].shape
    assert out_s.shape == inputs[1].shape
    assert torch.isfinite(out_u).all() and torch.isfinite(out_s).all()


def test_determinism(model, inputs, base_out):
    with torch.no_grad():
        out_u2, out_s2 = model(*inputs)
    assert torch.equal(out_u2, base_out[0])
    assert torch.equal(out_s2, base_out[1])


@pytest.mark.xfail(
    reason="equivariance is not an acceptance criterion; at Swin's window "
           "granularity this grid has no non-trivial equivariant roll "
           "(see docstring / CLAUDE.md)",
    strict=False)
@pytest.mark.parametrize("k", [12, 24, 60])
def test_theta_equivariance(model, inputs, base_out, k):
    u, s = inputs
    with torch.no_grad():
        out_u, out_s = model(torch.roll(u, k, dims=AX_U), torch.roll(s, k, dims=AX_S))
    exp_u = torch.roll(base_out[0], k, dims=AX_U)
    exp_s = torch.roll(base_out[1], k, dims=AX_S)
    du = (out_u - exp_u).abs().max().item()
    ds = (out_s - exp_s).abs().max().item()
    assert du < TOL and ds < TOL, (
        f"theta roll k={k} is not equivariant: upper_max_diff={du:.3e}, "
        f"surf_max_diff={ds:.3e} (tolerance {TOL:.0e})"
    )
