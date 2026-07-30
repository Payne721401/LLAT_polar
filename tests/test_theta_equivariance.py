"""
PanguPolarModel 測試(smoke / determinism / θ-roll 等變性診斷)。

跑法(cwd 在 regional_model/DLAMPty_polar):
    python -m pytest tests/ -v

正式測試(會過):
    test_smoke        —— 模型能 forward、形狀正確、無 NaN
    test_determinism  —— eval 下同輸入兩次完全相同

診斷(xfail,不是紅色關卡):
    test_theta_equivariance —— θ 旋轉等變。**刻意標 xfail**,原因(2026-07 實測釐清):
      1. 等變**不是**極座標的驗收標準。真需求是「θ 接縫連續」,而這個 transformer
         用非重疊 patch embed,實測**天生抗接縫**(循環/零填充、拆/留 mask 全都無縫)。
      2. Swin 視窗注意力只對「roll 整數個**視窗**」等變,不是整數個 patch。真正粒度是
         k = patch_θ × window_θ,且需 window_θ 偶數 + coarse 單一視窗。學姊的視窗
         (15,10)不滿足 → 圈內**沒有任何**非平凡 k 等變(只有 k=180 整圈)。
      3. 詳見 CLAUDE.md「測試指令」。此測試留著只當數值診斷,不當通過門檻。
    （若之後改成可等變的網格/視窗,可拿掉 xfail 並把 k 換成該設計的真實粒度。）
"""
import pytest
import torch

from models.pangu_polar import PanguPolarModel

# 依 config.yaml 的參數
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
AX_U, AX_S = 3, 2   # (B,Z,R,Θ,C) / (B,R,Θ,C) 的 Θ 軸
TOL = 1e-5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = PanguPolarModel(**CFG)
    m.eval()   # 關掉 DropPath
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


@pytest.mark.xfail(reason="等變非驗收標準；Swin 視窗粒度下此網格圈內無非平凡等變 roll（見 docstring / CLAUDE.md）",
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
        f"θ-roll k={k} 不等變: upper_max_diff={du:.3e}, surf_max_diff={ds:.3e} "
        f"(門檻 {TOL:.0e})"
    )
