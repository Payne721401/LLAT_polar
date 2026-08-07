"""Trainability / structural-soundness checks that need no training.

- test_backward_grad_coverage: backward runs and every parameter receives a
  finite gradient. This catches detached components and NaN/Inf gradients, which
  are properties of the architecture rather than of any particular weights.
- test_dead_bias_landmine: records landmine 3 - earth_specific_bias is currently
  not a Parameter, is all zeros, and never trains. Flip this assertion once it
  is fixed.
"""
import pytest
import torch

from models.pangu_polar import PanguPolarModel

CFG = dict(
    data_spatial_shape=(13, 201, 180), upper_vars=6, surface_vars=20,
    depths=[2, 6], heads=[6, 12], embed_dim=192,
    patch_shape=(2, 8, 6), window_size1=(2, 10, 15), window_size2=(2, 8, 10),
)


def test_backward_grad_coverage():
    torch.manual_seed(0)
    m = PanguPolarModel(**CFG)
    m.eval()   # disable DropPath so every path is live and coverage is deterministic
    g = torch.Generator().manual_seed(1)
    u = torch.randn(1, 13, 201, 180, 6, generator=g)
    s = torch.randn(1, 201, 180, 20, generator=g)

    out_u, out_s = m(u, s)
    loss = out_u.abs().mean() + out_s.abs().mean()
    loss.backward()

    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [n for n, p in m.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not no_grad, f"parameters received no gradient (possibly detached): {no_grad[:10]}"
    assert not nonfinite, f"non-finite gradients (NaN/Inf): {nonfinite[:10]}"


def test_dead_bias_landmine():
    """Landmine 3: the relative position bias is dead (not a Parameter).

    Flip this test once it is fixed.
    """
    m = PanguPolarModel(**CFG)
    bias_params = [n for n, _ in m.named_parameters() if "earth_specific_bias" in n]
    assert bias_params == [], (
        "earth_specific_bias is now a Parameter - landmine 3 has been fixed, so "
        "change this test to assert that the bias exists and has the right shape"
    )
