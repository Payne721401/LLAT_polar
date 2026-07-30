"""不需訓練就能測的『可訓練性 / 架構健全性』檢查。

- test_backward_grad_coverage:backward 跑得動、每個參數都收到有限梯度
  (抓「斷開的元件」「NaN/Inf 梯度」——這些跟權重值無關,是架構性質)。
- test_dead_bias_landmine:記錄地雷 3(earth_specific_bias 目前非 Parameter、
  全零、不訓練)。修好後請把這個測試反過來 assert。
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
    m.eval()   # 關 DropPath,讓所有路徑都在、梯度覆蓋確定
    g = torch.Generator().manual_seed(1)
    u = torch.randn(1, 13, 201, 180, 6, generator=g)
    s = torch.randn(1, 201, 180, 20, generator=g)

    out_u, out_s = m(u, s)
    loss = out_u.abs().mean() + out_s.abs().mean()
    loss.backward()

    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [n for n, p in m.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not no_grad, f"未收到梯度的參數(可能斷開): {no_grad[:10]}"
    assert not nonfinite, f"梯度非有限(NaN/Inf): {nonfinite[:10]}"


def test_dead_bias_landmine():
    """地雷 3:相對位置 bias 目前是死的(非 Parameter)。修好後請 flip 這個測試。"""
    m = PanguPolarModel(**CFG)
    bias_params = [n for n, _ in m.named_parameters() if "earth_specific_bias" in n]
    assert bias_params == [], (
        "earth_specific_bias 已成為 Parameter —— 地雷 3 已修,請把本測試改成"
        "『assert bias 存在且形狀正確』"
    )
