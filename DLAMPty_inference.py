import pandas as pd
import numpy as np
import yaml, sys, os
import xarray as xr
import onnxruntime as ort
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.data_processor import lonlat_uniformizer, recalc_additional_np,calc_additional_vars,to_xarray

import numpy as np
from scipy.ndimage import map_coordinates


def latlon_to_polar(
    data,
    R=201,
    Theta=180,
    r_max=40.0,
    center_xy=(40.0, 40.0),
    mode="bilinear",
    padding_mode="constant",
    return_theta_deg=True,
):
    """
    Input:
        (Y, X, V) or (A, Y, X, V)

    Output:
        (R, Theta, V) or (A, R, Theta, V)
    """
    data = np.asarray(data)

    if data.ndim == 3:
        data = data[None, ...]
        squeeze = True
    elif data.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D data, got {data.shape}")

    A, Y, X, V = data.shape
    cx, cy = center_xy

    r = np.linspace(0, r_max, R)
    theta = np.linspace(0, 2 * np.pi, Theta, endpoint=False)

    # Shape: (R, Theta)
    rr, tt = np.meshgrid(r, theta, indexing="ij")

    x = cx + rr * np.cos(tt)
    y = cy + rr * np.sin(tt)

    coordinates = np.array([y, x])

    order = 1 if mode == "bilinear" else 0

    # Output shape: (A, R, Theta, V)
    polar = np.empty((A, R, Theta, V), dtype=data.dtype)

    for a in range(A):
        for v in range(V):
            polar[a, :, :, v] = map_coordinates(
                data[a, :, :, v],
                coordinates,
                order=order,
                mode=padding_mode,
                cval=0.0,
            )

    if squeeze:
        polar = polar[0]

    if return_theta_deg:
        theta = np.rad2deg(theta)

    return polar, r, theta

import numpy as np
from scipy.ndimage import map_coordinates


def polar_to_latlon(
    polar,
    output_shape=(81, 81),
    r_max=40.0,
    center_xy=(40.0, 40.0),
    mode="bilinear",
    fill_value=0.0,
):
    """
    Convert polar-coordinate data back to a Cartesian grid.

    Parameters
    ----------
    polar : np.ndarray
        Polar data with shape:
            (R, Theta, V)
        or:
            (A, R, Theta, V)

    output_shape : tuple[int, int]
        Output Cartesian-grid shape as (Y, X).

    r_max : float
        Maximum radius represented by the polar grid.

    center_xy : tuple[float, float]
        Center location in Cartesian-grid coordinates as (cx, cy).

    mode : {"bilinear", "nearest"}
        Interpolation method.

    fill_value : float
        Value assigned to locations outside r_max.

    Returns
    -------
    cartesian : np.ndarray
        Cartesian data with shape:
            (Y, X, V)
        or:
            (A, Y, X, V)
    """
    polar = np.asarray(polar)

    if polar.ndim == 3:
        polar = polar[None, ...]
        squeeze = True
    elif polar.ndim == 4:
        squeeze = False
    else:
        raise ValueError(
            f"Expected polar data with 3 or 4 dimensions, got {polar.shape}"
        )

    # New dimension order: (A, R, Theta, V)
    A, R, Theta, V = polar.shape

    Y, X = output_shape
    cx, cy = center_xy

    order = 1 if mode == "bilinear" else 0

    # Cartesian-grid coordinates
    yy, xx = np.meshgrid(
        np.arange(Y, dtype=np.float64),
        np.arange(X, dtype=np.float64),
        indexing="ij",
    )

    dx = xx - cx
    dy = yy - cy

    # Cartesian coordinates -> physical polar coordinates
    radius = np.sqrt(dx**2 + dy**2)
    theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)

    # Physical polar coordinates -> polar-array indices
    radius_index = radius / r_max * (R - 1)
    theta_index = theta / (2.0 * np.pi) * Theta

    # Theta is now axis=2.
    # Append theta=0 data to the end for periodic interpolation.
    polar_periodic = np.concatenate(
        [polar, polar[:, :, :1, :]],
        axis=2,
    )

    # Coordinate order must match the input array:
    # polar_periodic[a, radius, theta, v]
    coordinates = np.array(
        [
            radius_index,
            theta_index,
        ]
    )

    output_dtype = np.result_type(polar.dtype, np.float32)

    cartesian = np.full(
        (A, Y, X, V),
        fill_value,
        dtype=output_dtype,
    )

    valid_mask = radius <= r_max

    for a in range(A):
        for v in range(V):
            interpolated = map_coordinates(
                polar_periodic[a, :, :, v],
                coordinates,
                order=order,
                mode="nearest",
            )

            cartesian[a, :, :, v] = np.where(
                valid_mask,
                interpolated,
                fill_value,
            )

    if squeeze:
        cartesian = cartesian[0]

    return cartesian

class DLAMPty_model:
    def __init__(self, model_path, root_dir=os.path.dirname(__file__), device=None, cpu_num=10 ):
        self.model_path = model_path
        # 明確指定 utf-8:open() 不帶 encoding 會用系統預設,在中文 Windows 上
        # 是 cp950,讀到含非 ASCII 註解的 yaml 會 UnicodeDecodeError。
        # 叢集(Linux/UTF-8)上不會發作,所以這是只在本機重現的可攜性問題。
        with open(self.model_path, encoding='utf-8') as f:
            self.model_setting = yaml.safe_load(f)
        self.root_dir = root_dir
        self.device = device
        self.cpu_num = cpu_num
        self.uniformize_lonlat = True
        self.specify_resolution = 0.25
        
        self.onnx_path = os.path.join(self.root_dir,self.model_setting['onnx_path'])
        self.stat_mean_file = os.path.join(self.root_dir,self.model_setting['stat_mean_file'])
        self.stat_std_file = os.path.join(self.root_dir,self.model_setting['stat_std_file'])
        self.ingest_space_info = self.model_setting['ingest_space_info'] #ingest_space_info=True, add `lon` and `lat` to the std or mean
        self.upper_variables = self.model_setting['upper_vars']
        self.surface_variables = self.model_setting['surface_vars']
        self.pressure_levels = self.model_setting['pressure_levels']
        self.upper_units = self.model_setting['upper_units']
        self.surface_units = self.model_setting['surface_units']

        self._setup_polar_grid()

    def _setup_polar_grid(self):
        """由 yaml 的 `polar` 區塊推導所有極座標參數(S2)。

        設計原則:**yaml 只放訓練 config.yaml 裡真實存在的三個值**,
        其餘一律推導。這樣改網格時只要改 yaml 的三行,不必動任何程式,
        也不可能出現「改了 R 卻忘了改 r_max」這種半套修改。

            data_spatial_shape [Z, R, Theta]   ← config.yaml: data.data_spatial_shape
            r_degree_max                       ← config.yaml: data.r_degree_max
            original_resolution                ← config.yaml: data.original_resolution

        推導出來的量:
            polar_R / polar_theta  極座標網格數
            r_max_px               半徑上限,換算成【格】(latlon_to_polar 用格)
            cartesian_n            笛卡兒域邊長,與訓練端 _trim_var 同一條公式
            center_xy              取樣中心 = 笛卡兒域正中央
        """
        polar = self.model_setting.get('polar')
        if polar is None:
            raise KeyError(
                f"{self.model_path} 缺少 `polar:` 區塊。極座標推論必須明確宣告網格,"
                "否則會沿用寫死的 R=201/Theta=180/r_max=40,對其他網格的模型"
                "會 shape mismatch(或更糟:形狀碰巧對得上但幾何錯誤)。"
            )

        shape = list(polar['data_spatial_shape'])
        if len(shape) != 3:
            raise ValueError(f"data_spatial_shape 應為 [Z, R, Theta],得到 {shape}")
        self.polar_shape = tuple(shape)
        self.polar_R, self.polar_theta = shape[1], shape[2]

        self.r_degree_max = float(polar['r_degree_max'])
        self.original_resolution = float(polar['original_resolution'])
        self.wind_representation = polar.get('wind_representation', 'uv')

        # 半徑上限換算成「格」。訓練端 datasets.py 寫的是 `r_degree_max * 4`,
        # 那個 4 是寫死的 1/0.25 —— 換 0.1° 資料就會錯。這裡用除法,正確且通用。
        self.r_max_px = self.r_degree_max / self.original_resolution

        # 笛卡兒域邊長。與訓練端 datasets._trim_var 的公式一致:
        #     int(r_degree_max * 2 / original_resolution) + 1
        self.cartesian_n = int(self.r_degree_max * 2 / self.original_resolution) + 1

        # 取樣中心 = 笛卡兒域正中央(TC 在 Lagrangian 網格上恆在此)
        half = (self.cartesian_n - 1) / 2.0
        self.center_xy = (half, half)

        # lonlat_uniformizer 重建座標軸時用的間距,就是來源解析度。
        # __init__ 裡原本寫死 0.25,和 original_resolution 是同一件事 —— 統一由此決定。
        self.specify_resolution = self.original_resolution

        if abs(self.r_max_px - half) > 1e-9:
            raise ValueError(
                f"半徑上限({self.r_max_px} 格)與笛卡兒域半寬({half} 格)不一致。"
                "極座標圓應恰好內接於方形域,請檢查 r_degree_max 與 original_resolution。"
            )

    def _stat_from_nc(self, nc_file_path: str, want_sfc: bool) -> np.ndarray:
        """
        Args:
            nc_file_path (str): Path to the netCDF file.
            want_sfc (bool): Whether to return the mean or standard deviation of the surface data.

        Returns:
            np.ndarray: Mean or standard deviation of the upper-air data or surface data.
        """
        nc_path = nc_file_path
        with xr.open_dataset(nc_path) as input_nc:
            var_list = self.upper_variables if not want_sfc else self.surface_variables
            stat_data = np.stack([input_nc[var].values for var in var_list], axis=-1)
            if want_sfc and len(stat_data.shape) == 4 or len(stat_data.shape) == 5:
                stat_data = stat_data.squeeze(-2)
            # when getting sfc stat and ingest_space_info=True, add `lon` and `lat` to the std or mean
            if want_sfc and self.ingest_space_info:
                # the stat data of lon and lat if we are loading std data
                c_lon, c_lat = (12, 12)
                # determine whether we are loading mean data by nc_file_path
                if nc_file_path.find('mean') != -1:
                    # we are loading mean data
                    c_lon, c_lat = (130, 15)
                shape = stat_data.shape[0:-1]
                lon = np.expand_dims(np.zeros(shape)+c_lon, axis=-1) 
                lat = np.expand_dims(np.zeros(shape)+c_lat, axis=-1) 
                stat_data = np.concatenate((stat_data, lon, lat), axis=-1)
            stat_data = np.swapaxes(stat_data, 0, 1)
        return stat_data
    
    def load_statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of mean and std of upper-air data and surface data.
        """
        self.upper_mean = self._stat_from_nc(self.stat_mean_file, False)
        self.upper_std = self._stat_from_nc(self.stat_std_file, False)
        self.surface_mean = self._stat_from_nc(self.stat_mean_file, True)
        self.surface_std = self._stat_from_nc(self.stat_std_file, True)

    def load_model(self):        
        cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
        ort_providers = [
            ("CUDAExecutionProvider", cuda_provider_options),
            "CPUExecutionProvider",
        ]
        if self.device=='cpu' or not os.path.exists("/proc/driver/nvidia/version"):
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.cpu_num 
            ort_providers.pop(0)
            self.device = 'cpu'
            model = ort.InferenceSession(self.onnx_path, sess_options=session_options, providers=ort_providers)
        else:
            model = ort.InferenceSession(self.onnx_path, providers=ort_providers)
        print(f"inference with {ort_providers}")
        return model
    
    def initialize(self):
        if self.wind_representation == 'vt_vr':
            raise NotImplementedError(
                f"{self.model_path} 宣告 wind_representation: vt_vr,但 polar_to_latlon "
                "目前只做內插、沒有做向量旋轉(S4)。切向/徑向風被當純量搬回經緯度網格,"
                "送進 FCNV2 的會是物理上錯誤的風場。實作反向旋轉之後再移除此檢查:\n"
                "    u = -vt*sin(theta) + vr*cos(theta)\n"
                "    v =  vt*cos(theta) + vr*sin(theta)"
            )
        self.model = self.load_model()
        self._check_onnx_shape()
        self.load_statistics()
        print(f"Model and weights are loaded. "
              f"polar grid (Z,R,Theta)={self.polar_shape}, "
              f"r_max={self.r_max_px:g} px ({self.r_degree_max:g} deg), "
              f"cartesian {self.cartesian_n}x{self.cartesian_n}")

    def _check_onnx_shape(self):
        """比對 yaml 宣告的極座標網格與 onnx 實際期望的輸入形狀。

        沒有這個檢查,網格不合會等到第一次 model.run() 才炸(訊息還很難讀);
        更糟的是若形狀碰巧相容但幾何不同(例如 R 對了但 r_degree_max 錯),
        永遠不會報錯,只會安靜地輸出錯誤結果。
        """
        want = {'input_upper': (self.polar_shape[0], self.polar_R, self.polar_theta,
                                len(self.upper_variables)),
                'input_surface': (self.polar_R, self.polar_theta,
                                  len(self.surface_variables)
                                  + (2 if self.ingest_space_info else 0))}
        for inp in self.model.get_inputs():
            if inp.name not in want:
                continue
            # onnx 的第 0 維是 batch,可能是動態的(字串);只比對其後各維
            got = tuple(d for d in inp.shape[1:] if isinstance(d, int))
            exp = want[inp.name]
            if len(got) == len(exp) and got != exp:
                raise ValueError(
                    f"{inp.name} 形狀不合:onnx 期望 {got},但 yaml 的 polar 區塊"
                    f"推導出 {exp}。請確認 {self.model_path} 的 data_spatial_shape / "
                    f"變數表與訓練時的 config.yaml 一致。")
        
    def normalize(self, upper_data, surface_data, reverse=False):
        if reverse:
            new_upper = upper_data * self.upper_std + self.upper_mean
            new_surface = surface_data * self.surface_std + self.surface_mean
        else:
            new_upper = (upper_data - self.upper_mean) / self.upper_std 
            new_surface = (surface_data - self.surface_mean) / self.surface_std
        return new_upper, new_surface
        
    def predict_one_step(self, input_upper, input_surface):
        
        # 形狀先驗證再轉換:笛卡兒域邊長由 r_degree_max/original_resolution 推導,
        # 對不上代表 IC 的裁切範圍與訓練時不同,再往下跑只會得到錯誤結果。
        n = self.cartesian_n
        if input_surface.shape[:2] != (n, n) or input_upper.shape[1:3] != (n, n):
            raise ValueError(
                f"輸入的笛卡兒網格應為 {n}x{n}(由 r_degree_max={self.r_degree_max:g} 度 / "
                f"original_resolution={self.original_resolution:g} 度推導),"
                f"實際拿到 upper {input_upper.shape}、surface {input_surface.shape}。")

        polar_kw = dict(R=self.polar_R, Theta=self.polar_theta,
                        r_max=self.r_max_px, center_xy=self.center_xy)
        input_upper, _, _ = latlon_to_polar(input_upper, **polar_kw)
        input_surface, _, _ = latlon_to_polar(input_surface, **polar_kw)
        input_upper,input_surface = self.normalize(input_upper,input_surface)
        input_upper = np.expand_dims(input_upper, axis=0)
        input_surface = np.expand_dims(input_surface, axis=0)
        
        ort_inputs = {
            "input_upper": input_upper.astype(np.float32),
            "input_surface": input_surface.astype(np.float32)
        }
        
        # Run inference
        ort_outputs = self.model.run(None,ort_inputs)
        output_upper = ort_outputs[0].squeeze()
        output_surface = ort_outputs[1].squeeze()

        # reverse back
        output_upper, output_surface = self.normalize(output_upper,output_surface,reverse=True)
        # ⚠️ 圓外(方形的四個角,佔 23.4%)沒有模型輸出,必須填 NaN 而非 0。
        #    填 0 的後果(B1→B2):lon 通道被 0 稀釋成 0.766 倍,
        #    lonlat_uniformizer 算出的中心從 130°E 變成 99.6°E ——
        #    耦合會把 LLAT 的場貼到全球網格偏西 3,300 km 的位置。
        #    NaN 不會傳染:latlon_to_polar 只取樣 r ≤ r_max(碰不到角落),
        #    LLAT→FCNV2 只讀 r < 7.5°,而 FCNV2→LLAT 反而會把角落補回來。
        back_kw = dict(output_shape=(self.cartesian_n, self.cartesian_n),
                       r_max=self.r_max_px, center_xy=self.center_xy,
                       mode="bilinear", fill_value=np.nan)
        output_upper = polar_to_latlon(output_upper, **back_kw)
        output_surface = polar_to_latlon(output_surface, **back_kw)

        # uniform lat lon
        if self.uniformize_lonlat:
            lon, lat = lonlat_uniformizer(
                output_surface[:, :, -2],
                output_surface[:, :, -1],
                self.uniformize_lonlat,
                self.specify_resolution,
            )
            # B6:這行原本縮排在 if 之外,uniformize_lonlat=False 時
            #     lon/lat 未定義會 NameError。移進區塊內。
            (output_surface[:, :, -2], output_surface[:, :, -1]) = np.meshgrid(lon, lat)

        return output_upper, output_surface
    
    def changing_additional_information(self, input_upper, input_surface, timestep):
        additionals = recalc_additional_np(
                input_upper, input_surface, timestep,
                self.upper_variables, self.surface_variables, self.upper_units, self.surface_units
            )

        for v in self.surface_variables:
            if v in additionals:
                i = self.surface_variables.index(v)
                input_surface[:, :, i] = additionals[v]

        for v in self.upper_variables:
            if v in additionals:
                i = self.upper_variables.index(v)
                input_upper[:, :, :, i] = additionals[v]
        
        return input_upper, input_surface
    
    def IC_from_xarray_to_npy(self, IC_dataset:xr.Dataset, additional_vars=False):
        if not additional_vars:
            print('It needs to calc additional vars.')
            IC_dataset = calc_additional_vars(IC_dataset, True)
        input_upper = np.stack([IC_dataset[var].values for var in self.upper_variables], axis=-1).squeeze()     
        input_surface = np.stack([IC_dataset[var].values for var in self.surface_variables], axis=-1).squeeze()
        lon, lat = np.meshgrid(IC_dataset.longitude, IC_dataset.latitude)
        input_surface = np.concatenate((input_surface, np.stack([lon, lat],axis=-1)), axis=-1)
        return input_upper, input_surface

    def data_to_xarray(self, upper_data, surface_data, timestep):
        DLAMPty_xr = to_xarray(upper_data, surface_data, self.upper_variables, self.surface_variables, self.upper_units, self.surface_units, self.pressure_levels)
        DLAMPty_xr = DLAMPty_xr.expand_dims(time=[pd.to_datetime(timestep)])
        return DLAMPty_xr
