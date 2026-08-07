"""極座標 ↔ 笛卡兒來回轉換的正確性(B1 / B2 / B6 迴歸測試)。

背景
----
LLAT 的 surface 最後兩個通道是 `lon` / `lat`,它們是**被預測的變數**,不是
座標軸 —— 因為網格跟著颱風跑(Lagrangian),「颱風移動」在實作上就是
「lon/lat 場整體平移」。所以下游是用 `lon/lat 場的平均` 當作颱風中心,
再據此決定要把 LLAT 的場貼到 FCNV2 全球網格的哪個位置。

B1 / B2 這對 bug 就是攻擊這條鏈:
  B1  `polar_to_latlon(fill_value=0.0)` —— 極座標圓裝不滿 81×81 方框,
      四個角落(1536 格 = 23.4%)被填成物理 0。
  B2  `lonlat_uniformizer` 用整個場的平均定位中心 ⇒ 被 0 稀釋成 0.766 倍
      ⇒ 130°E 被算成 99.6°E,偏 3,300 km,而且會逐步回饋惡化。

修法是 `fill_value=np.nan` + 全面 `nanmean`。本檔把「改壞了就會紅」的
性質全部鎖住。

注意:這裡直接測 `DLAMPty_inference` 內的兩個轉換函式與 `lonlat_uniformizer`,
不需要 onnx 權重。
"""
import numpy as np
import pytest

from DLAMPty_inference import latlon_to_polar, polar_to_latlon
from utils.data_processor import lonlat_uniformizer

# 與 predict_one_step 內的呼叫一致
R, THETA, R_MAX = 201, 180, 40.0
N = 81
CENTER = (40.0, 40.0)
RES = 0.25
LON0, LAT0 = 130.0, 15.0


def make_lonlat():
    """造一個中心在 (130°E, 15°N)、0.25° 的乾淨 lon/lat 場,形狀 (81, 81, 2)。

    緯度**由北往南遞減**,與 ERA5 及本專案的資料一致 —— 方向測試靠這個。
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
# 幾何:圓裝不滿方,這是問題的源頭
# --------------------------------------------------------------------------

def test_circle_does_not_fill_square():
    """r_max=40 的圓只蓋住 81×81 的 76.6%,四角必然沒有資料。

    這不是 bug 而是幾何事實,但它是 B1 的前提 —— 鎖住它,才能保證
    「角落要怎麼填」這個問題不會被誰不小心改掉 r_max 而消失。
    """
    yy, xx = np.meshgrid(np.arange(N) - 40.0, np.arange(N) - 40.0, indexing="ij")
    inside = np.hypot(xx, yy) <= R_MAX
    assert inside.sum() == 5025
    assert (~inside).sum() == 1536
    assert 0.76 < inside.mean() < 0.77
    # 角落離中心 √(40²+40²) = 56.6 格,遠在半徑 40 之外
    assert np.hypot(40.0, 40.0) > R_MAX


# --------------------------------------------------------------------------
# B1:圓外必須是 NaN,圓內必須完好
# --------------------------------------------------------------------------

def test_outside_is_nan_and_inside_is_exact():
    back = roundtrip(make_lonlat(), np.nan)
    yy, xx = np.meshgrid(np.arange(N) - 40.0, np.arange(N) - 40.0, indexing="ij")
    inside = np.hypot(xx, yy) <= R_MAX

    assert np.isnan(back[~inside]).all(), "圓外應該全部是 NaN"
    assert not np.isnan(back[inside]).any(), "圓內不該出現 NaN"

    truth = make_lonlat()
    # 圓內是雙線性內插,不會逐位元相同,但誤差應該遠小於一格(0.25°)
    assert np.abs(back[inside] - truth[inside]).max() < 0.05


# --------------------------------------------------------------------------
# B2:中心必須算對,而且座標軸方向不能反
# --------------------------------------------------------------------------

def test_center_survives_roundtrip():
    """一次來回之後,推得的中心必須還在原地(誤差 < 0.01°)。"""
    back = roundtrip(make_lonlat(), np.nan)
    lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)

    assert np.nanmean(lon) == pytest.approx(LON0, abs=0.01)
    assert np.nanmean(lat) == pytest.approx(LAT0, abs=0.01)


def test_axis_direction_preserved():
    """經度遞增、緯度遞減。

    這條看似瑣碎,卻是 nanmean 修正裡**最容易漏掉**的一環:
    `lonlat_uniformizer` 用 `np.diff(...).mean()` 的正負號決定軸的方向。
    若那兩個 mean 沒改成 nanmean,結果是 NaN,而
    `specify_resolution * lon_res >= 0` 對 NaN 恆為 False
    ⇒ 經度軸被整個反向,且不會有任何錯誤訊息。
    """
    back = roundtrip(make_lonlat(), np.nan)
    lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)

    assert lon[1] - lon[0] == pytest.approx(+RES, abs=1e-9), "經度必須遞增"
    assert lat[1] - lat[0] == pytest.approx(-RES, abs=1e-9), "緯度必須遞減"


def test_center_stable_over_many_steps():
    """自迴歸情境:把上一步算出的軸寫回場,再跑一次(模擬 predict_one_step)。

    B2 的殺傷力在於**會複利** —— 填 0 時每步乘 0.766,五步後
    130°E 會掉到 34°E。修好之後應該完全不漂移。
    """
    cur = make_lonlat()
    for _ in range(5):
        back = roundtrip(cur, np.nan)
        lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)
        cur = np.stack(np.meshgrid(lon, lat), axis=-1)

    assert np.nanmean(cur[..., 0]) == pytest.approx(LON0, abs=0.01)
    assert np.nanmean(cur[..., 1]) == pytest.approx(LAT0, abs=0.01)


# --------------------------------------------------------------------------
# 負向對照:證明這組測試真的抓得到 B1/B2(否則它可能只是恆真)
# --------------------------------------------------------------------------

def test_fill_zero_reproduces_the_bug():
    """故意用舊的 fill_value=0.0,必須重現 ×0.766 的塌縮。

    沒有這條,上面幾個測試無法證明自己有鑑別力。
    """
    back = roundtrip(make_lonlat(), 0.0)
    lon, lat = lonlat_uniformizer(back[:, :, 0], back[:, :, 1], True, RES)

    ratio = 5025 / (N * N)                      # 圓佔方的面積比 ≈ 0.766
    assert np.nanmean(lon) == pytest.approx(LON0 * ratio, rel=0.02)
    assert np.nanmean(lat) == pytest.approx(LAT0 * ratio, rel=0.02)
    # 換算成距離:經度偏掉超過 3,000 km
    err_km = (LON0 - np.nanmean(lon)) * 111 * np.cos(np.deg2rad(LAT0))
    assert err_km > 3000


# --------------------------------------------------------------------------
# nanmean 的向後相容:笛卡兒路徑(無 NaN)行為不能變
# --------------------------------------------------------------------------

def test_cartesian_path_unchanged():
    """沒有 NaN 時 nanmean ≡ mean,原本的笛卡兒推論路徑不受影響。"""
    sfc = make_lonlat()
    lon, lat = lonlat_uniformizer(sfc[:, :, 0], sfc[:, :, 1], True, RES)

    assert np.nanmean(lon) == pytest.approx(LON0, abs=1e-9)
    assert np.nanmean(lat) == pytest.approx(LAT0, abs=1e-9)
    assert lon[1] - lon[0] == pytest.approx(+RES, abs=1e-9)
    assert lat[1] - lat[0] == pytest.approx(-RES, abs=1e-9)
