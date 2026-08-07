"""推論端的極座標網格參數化(S2)與 vt/vr yaml(S3)。

設計意圖
--------
`onnx/*.yaml` 的 `polar:` 區塊只放**訓練 config.yaml 裡真實存在的三個值**:

    data_spatial_shape [Z, R, Theta]
    r_degree_max          (度)
    original_resolution   (度/格)

其餘(半徑上限的格數、笛卡兒域邊長、取樣中心、uniformizer 的間距)全部推導。
這樣換網格只要改 yaml 三行,不可能出現「改了 R 卻忘了改 r_max」的半套修改
—— 那正是 S2 之前的狀態(R/Theta/r_max 寫死在 predict_one_step 裡)。

本檔不需要 .onnx 權重:只建構物件、不呼叫 initialize()。
"""
import os

import pytest
import yaml

from DLAMPty_inference import DLAMPty_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_UV = os.path.join(ROOT, "onnx", "LLAT_polar_v1.yaml")
YAML_VTVR = os.path.join(ROOT, "onnx", "LLAT_polar_vtvr_v1.yaml")


def write_yaml(tmp_path, **polar_overrides):
    """以 vt/vr yaml 為底,改寫 polar 區塊,產生一份臨時 yaml。"""
    cfg = yaml.safe_load(open(YAML_VTVR, encoding="utf-8"))
    cfg["polar"].update(polar_overrides)
    p = tmp_path / "tmp.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------
# S3:vt/vr yaml 的內容
# --------------------------------------------------------------------------

def test_vtvr_yaml_matches_training_config():
    """yaml 的變數表與網格必須和訓練用 config.yaml 一致。

    這是整條推論鏈最容易靜默出錯的地方:變數順序錯一格,模型會把濕度
    當成溫度讀,而且完全不會報錯。
    """
    cfg = yaml.safe_load(open(YAML_VTVR, encoding="utf-8"))
    train = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))

    assert cfg["upper_vars"] == train["data"]["upper_variables"]
    assert cfg["surface_vars"] == train["data"]["surface_variables"]
    assert cfg["polar"]["data_spatial_shape"] == train["data"]["data_spatial_shape"]
    assert cfg["polar"]["r_degree_max"] == train["data"]["r_degree_max"]
    assert cfg["polar"]["original_resolution"] == train["data"]["original_resolution"]
    # 統計檔也必須是同一份,否則標準化的基準不同
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
# S2:參數推導
# --------------------------------------------------------------------------

def test_derived_grid_uv():
    """學姊的 baseline:R=201、Δr 0.05°,但笛卡兒域仍是 81×81。"""
    m = DLAMPty_model(YAML_UV, root_dir=ROOT)
    assert (m.polar_R, m.polar_theta) == (201, 180)
    assert m.r_max_px == 40.0            # 10 度 / 0.25 度每格
    assert m.cartesian_n == 81           # 10*2/0.25 + 1
    assert m.center_xy == (40.0, 40.0)
    assert m.specify_resolution == 0.25
    assert m.wind_representation == "uv"


def test_derived_grid_vtvr():
    """我的模型:R=41(Δr 對齊來源解析度),其餘幾何不變。"""
    m = DLAMPty_model(YAML_VTVR, root_dir=ROOT)
    assert (m.polar_R, m.polar_theta) == (41, 180)
    assert m.r_max_px == 40.0
    assert m.cartesian_n == 81
    assert m.center_xy == (40.0, 40.0)


def test_grid_follows_yaml_without_touching_code(tmp_path):
    """改 yaml 的三個值,推導出來的網格就跟著變 —— 這是 S2 的重點。

    這裡故意用未來要跑的 0.1° 高解析情境:訓練端 datasets.py 寫的是
    `r_max = r_degree_max * 4`(把 1/0.25 寫死),換成 0.1° 就會算成 40 而非 100。
    推論端用除法,所以這條會過;若哪天有人把它改回乘法,這條立刻紅。
    """
    p = write_yaml(tmp_path, data_spatial_shape=[13, 100, 360],
                   r_degree_max=10, original_resolution=0.1,
                   wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)
    assert (m.polar_R, m.polar_theta) == (100, 360)
    assert m.r_max_px == 100.0           # 10 / 0.1,不是 10*4
    assert m.cartesian_n == 201          # 10*2/0.1 + 1
    assert m.center_xy == (100.0, 100.0)
    assert m.specify_resolution == 0.1


# --------------------------------------------------------------------------
# 防呆:設定寫錯要當場報錯,不能安靜地跑出錯誤結果
# --------------------------------------------------------------------------

def test_missing_polar_block_raises(tmp_path):
    cfg = yaml.safe_load(open(YAML_VTVR, encoding="utf-8"))
    del cfg["polar"]
    p = tmp_path / "no_polar.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(KeyError, match="polar"):
        DLAMPty_model(str(p), root_dir=ROOT)


def test_non_integer_radius_raises(tmp_path):
    """半徑換算成格數若不是整數,就報錯。

    `cartesian_n = int(2*r_degree_max/original_resolution) + 1` 裡的 `int()`
    會無條件捨去。只要 `2*r_degree_max/original_resolution` 不是整數,
    笛卡兒域的半寬就和極座標半徑對不上 —— 圓不再內接於方,取樣幾何
    與訓練時不同,但形狀仍然合法、模型照跑、不會有任何錯誤訊息。

    例:r_degree_max=10 搭 0.3° ⇒ 半徑 33.33 格,但域半寬只有 33 格。
    (合法的組合如 10/0.25 → 40 格、7/0.25 → 28 格,都會通過。)
    """
    p = write_yaml(tmp_path, r_degree_max=10, original_resolution=0.3,
                   wind_representation="uv")
    with pytest.raises(ValueError, match="內接"):
        DLAMPty_model(p, root_dir=ROOT)


def test_other_valid_resolutions_accepted(tmp_path):
    """反向確認上一條不是恆真:整除的組合必須通過。"""
    p = write_yaml(tmp_path, data_spatial_shape=[13, 41, 180],
                   r_degree_max=7, original_resolution=0.25,
                   wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)
    assert m.r_max_px == 28.0
    assert m.cartesian_n == 57
    assert m.center_xy == (28.0, 28.0)


# --------------------------------------------------------------------------
# 接線:predict_one_step 必須真的用推導出來的值
# --------------------------------------------------------------------------

class _FakeSession:
    """假的 onnxruntime session:記下實際收到的形狀,回傳同形狀的零。

    只驗「餵給模型的極座標網格對不對」,不需要真的權重。
    """

    def __init__(self, z, r, theta, n_upper, n_sfc):
        self.shape = (z, r, theta, n_upper)
        self.sfc_shape = (r, theta, n_sfc)
        self.seen = None

    def run(self, _, inputs):
        self.seen = {k: v.shape for k, v in inputs.items()}
        import numpy as _np
        return [_np.zeros((1,) + self.shape, dtype=_np.float32),
                _np.zeros((1,) + self.sfc_shape, dtype=_np.float32)]


def test_predict_one_step_uses_derived_grid(tmp_path):
    """把 yaml 的 R 改成一個特別的數,模型收到的形狀就必須跟著變。

    這條才真正鎖住 S2。只檢查 `m.polar_R` 這類屬性是不夠的 ——
    程式完全可以推導出正確的值卻在 predict_one_step 裡繼續用寫死的
    R=201,屬性測試一樣全綠。實測過:還原成硬編碼時,只有這條會紅。
    """
    import numpy as np

    R_ODD, THETA_ODD = 41, 180
    p = write_yaml(tmp_path, data_spatial_shape=[13, R_ODD, THETA_ODD],
                   r_degree_max=10, original_resolution=0.25,
                   wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)

    n_sfc = len(m.surface_variables) + 2          # ingest_space_info 追加 lon/lat
    m.model = _FakeSession(13, R_ODD, THETA_ODD, len(m.upper_variables), n_sfc)
    # 標準化在這裡不是重點,設成恆等
    m.upper_mean = m.surface_mean = 0.0
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
    # 轉回來的形狀要回到笛卡兒域
    assert out_u.shape == (13, n, n, len(m.upper_variables))
    assert out_s.shape == (n, n, n_sfc)


def test_predict_one_step_rejects_wrong_input_size(tmp_path):
    """IC 的裁切範圍和 yaml 推導的域大小不符時,要當場報錯。"""
    import numpy as np

    p = write_yaml(tmp_path, wind_representation="uv")
    m = DLAMPty_model(p, root_dir=ROOT)
    bad = 61                                       # 應為 81
    with pytest.raises(ValueError, match="笛卡兒網格"):
        m.predict_one_step(np.zeros((13, bad, bad, len(m.upper_variables))),
                           np.zeros((bad, bad, len(m.surface_variables) + 2)))


def test_vtvr_blocks_until_rotation_implemented():
    """S4 未實作前,載入 vt/vr 模型必須明確拒絕,而不是輸出錯誤的風場。

    polar_to_latlon 只做內插。vt/vr 是座標相依的向量,直接搬回經緯度網格
    得到的不是 u/v;送進 FCNV2 會是物理上錯誤的風。
    """
    m = DLAMPty_model(YAML_VTVR, root_dir=ROOT)
    with pytest.raises(NotImplementedError, match="vt_vr"):
        m.initialize()
