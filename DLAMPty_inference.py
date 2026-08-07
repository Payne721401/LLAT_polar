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


# ---------------------------------------------------------------------------
# Tangential / radial wind (S4)
# ---------------------------------------------------------------------------
# The model works in vt/vr while FCNV2, the saved output and the plotting code
# all use u/v, so the wrapper has to rotate on the way in and on the way out.
#
# Rotation is done in POLAR space, where theta is exact per column. Doing it
# after polar_to_latlon would mean interpolating vt/vr, whose meaning depends on
# theta, across neighbouring azimuths - correct only in the limit of fine dtheta.
# Interpolating u/v instead is exact everywhere, so rotate first, interpolate
# second.
#
# The sign convention cannot be derived from first principles here. The dataset
# ships vt/vr precomputed and this project never produced them, and there are
# several independent places to get a sign wrong: whether the row index runs
# north-to-south (ERA5 latitude is usually descending, which flips the sense of
# theta), whether vt is positive counter-clockwise, and whether vr is positive
# outward. So the convention is declared in the model yaml and determined
# empirically by tools/verify_vtvr_convention.py.
#
# Parameterisation. With theta the azimuth used by latlon_to_polar
# (x = cx + r*cos(theta), y = cy + r*sin(theta)) the forward transform is
#
#     vt = p*u*sin(theta) + q*v*cos(theta)
#     vr = s*u*cos(theta) + t*v*sin(theta)
#
# with p, q, s, t in {+1, -1}. Orthogonality (rows of the 2x2 matrix must be
# perpendicular for every theta) requires p*s == -q*t, which leaves 8 valid
# combinations. Because the matrix is orthogonal its inverse is its transpose:
#
#     u = p*vt*sin(theta) + s*vr*cos(theta)
#     v = q*vt*cos(theta) + t*vr*sin(theta)
WIND_CONVENTIONS = {
    # name: (p, q, s, t)
    'ccw_outward':      (-1,  1,  1,  1),   # vt CCW positive, vr outward positive
    'ccw_inward':       (-1,  1, -1, -1),
    'cw_outward':       ( 1, -1,  1,  1),
    'cw_inward':        ( 1, -1, -1, -1),
    # The four below differ by the sense of theta itself, which is what a
    # north-to-south row ordering produces.
    'ccw_outward_flip': ( 1,  1,  1, -1),
    'ccw_inward_flip':  ( 1,  1, -1,  1),
    'cw_outward_flip':  (-1, -1,  1, -1),
    'cw_inward_flip':   (-1, -1, -1,  1),
}


def _check_orthogonal(coeffs):
    p, q, s, t = coeffs
    if p * s != -q * t:
        raise ValueError(
            f"wind convention {coeffs} is not orthogonal (needs p*s == -q*t); "
            "it would not be invertible and would not preserve wind speed.")


def rotate_polar_wind(polar, theta_rad, i_a, i_b, convention, inverse):
    """Rotate a wind pair in place inside a polar array.

    Args:
        polar: (..., R, Theta, V) array; modified in place.
        theta_rad: (Theta,) azimuths, as returned by latlon_to_polar.
        i_a, i_b: channel indices. Forward they are (u, v); inverse (vt, vr).
        convention: key of WIND_CONVENTIONS.
        inverse: False for u,v -> vt,vr; True for vt,vr -> u,v.

    Returns:
        The same array, for chaining.
    """
    p, q, s, t = WIND_CONVENTIONS[convention]
    _check_orthogonal((p, q, s, t))
    sin_t = np.sin(theta_rad)[None, :]          # broadcast over R
    cos_t = np.cos(theta_rad)[None, :]

    a = polar[..., i_a].copy()
    b = polar[..., i_b].copy()
    if inverse:                                  # (vt, vr) -> (u, v)
        polar[..., i_a] = p * a * sin_t + s * b * cos_t
        polar[..., i_b] = q * a * cos_t + t * b * sin_t
    else:                                        # (u, v) -> (vt, vr)
        polar[..., i_a] = p * a * sin_t + q * b * cos_t
        polar[..., i_b] = s * a * cos_t + t * b * sin_t
    return polar


# Model-internal name -> the name the rest of the world uses.
_WIND_ALIAS = {'vt': 'u', 'vr': 'v', 'vt10': 'u10', 'vr10': 'v10'}


class DLAMPty_model:
    def __init__(self, model_path, root_dir=os.path.dirname(__file__), device=None, cpu_num=10 ):
        self.model_path = model_path
        # Explicit utf-8: a bare open() uses the platform default, which is
        # cp950 on a Chinese Windows locale and raises UnicodeDecodeError on a
        # yaml containing non-ASCII comments. Linux/UTF-8 clusters never hit it,
        # so this only reproduces locally.
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
        """Derive every polar-grid quantity from the yaml `polar` block (S2).

        Design rule: the yaml holds ONLY the three values that actually exist in
        the training config.yaml; everything else is derived. Switching grids is
        then a three-line yaml edit, and a half-done change (new R, stale r_max)
        cannot be expressed.

            data_spatial_shape [Z, R, Theta]   <- config.yaml data.data_spatial_shape
            r_degree_max                       <- config.yaml data.r_degree_max
            original_resolution                <- config.yaml data.original_resolution

        Derived:
            polar_R / polar_theta  polar grid counts
            r_max_px               maximum radius in CELLS (latlon_to_polar's unit)
            cartesian_n            Cartesian domain size, same formula as _trim_var
            center_xy              sampling centre = middle of the Cartesian domain
        """
        polar = self.model_setting.get('polar')
        if polar is None:
            raise KeyError(
                f"{self.model_path} has no `polar:` block. The polar grid must be "
                "declared explicitly; otherwise the old hardcoded "
                "R=201/Theta=180/r_max=40 applies, which either mismatches shapes "
                "or - worse - happens to fit while the geometry is wrong."
            )

        shape = list(polar['data_spatial_shape'])
        if len(shape) != 3:
            raise ValueError(f"data_spatial_shape must be [Z, R, Theta], got {shape}")
        self.polar_shape = tuple(shape)
        self.polar_R, self.polar_theta = shape[1], shape[2]

        self.r_degree_max = float(polar['r_degree_max'])
        self.original_resolution = float(polar['original_resolution'])
        self.wind_representation = polar.get('wind_representation', 'uv')

        # Maximum radius expressed in CELLS. The training side used to write
        # `r_degree_max * 4`, hardcoding 1/0.25; that breaks on the planned
        # 0.1-degree data. Dividing by the resolution is correct for any grid.
        self.r_max_px = self.r_degree_max / self.original_resolution

        # Cartesian domain size, same formula as datasets._trim_var:
        #     int(r_degree_max * 2 / original_resolution) + 1
        self.cartesian_n = int(self.r_degree_max * 2 / self.original_resolution) + 1

        # Sampling centre = middle of the Cartesian domain. On the Lagrangian
        # grid the TC always sits there by construction.
        half = (self.cartesian_n - 1) / 2.0
        self.center_xy = (half, half)

        # Spacing lonlat_uniformizer uses when rebuilding the axes: this is just
        # the source resolution. __init__ hardcoded 0.25, which is the same thing;
        # decide it here so there is one source of truth.
        self.specify_resolution = self.original_resolution

        if abs(self.r_max_px - half) > 1e-9:
            raise ValueError(
                f"Radius ({self.r_max_px} cells) does not match the Cartesian "
                f"half-width ({half} cells). The polar disc must be inscribed in "
                "the square domain; check r_degree_max and original_resolution."
            )

        self._setup_wind()

    def _setup_wind(self):
        """Work out the vt/vr <-> u/v rotation this model needs (S4).

        The yaml variable lists describe what the MODEL consumes. Everything
        outside this wrapper - the FCNV2 coupling, the saved npy, the plotting -
        speaks u/v, so the public interface is u/v and the rotation happens
        inside predict_one_step. `*_variables_external` are the names that go
        with the arrays callers hand in and get back; `*_variables` stay as the
        model's own names because the normalisation statistics are keyed on them.
        """
        self.wind_convention = (self.model_setting.get('polar') or {}).get('wind_convention')

        self.upper_variables_external = [_WIND_ALIAS.get(v, v) for v in self.upper_variables]
        self.surface_variables_external = [_WIND_ALIAS.get(v, v) for v in self.surface_variables]

        self._wind_idx_upper = None
        self._wind_idx_surface = None
        if self.wind_representation != 'vt_vr':
            return

        def pair(names, a, b, where):
            if a not in names or b not in names:
                raise ValueError(
                    f"wind_representation is vt_vr but {where} lacks {a}/{b}: {names}")
            return names.index(a), names.index(b)

        self._wind_idx_upper = pair(self.upper_variables, 'vt', 'vr', 'upper_vars')
        self._wind_idx_surface = pair(self.surface_variables, 'vt10', 'vr10', 'surface_vars')

        if self.wind_convention is not None and self.wind_convention not in WIND_CONVENTIONS:
            raise ValueError(
                f"unknown wind_convention {self.wind_convention!r}; "
                f"expected one of {sorted(WIND_CONVENTIONS)}")

    def _rotate_wind(self, upper, surface, theta_rad, inverse):
        """Apply the rotation to both the upper and the surface wind pair."""
        rotate_polar_wind(upper, theta_rad, *self._wind_idx_upper,
                          self.wind_convention, inverse)
        rotate_polar_wind(surface, theta_rad, *self._wind_idx_surface,
                          self.wind_convention, inverse)
        return upper, surface

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
        if self.wind_representation == 'vt_vr' and self.wind_convention is None:
            raise ValueError(
                f"{self.model_path} declares wind_representation: vt_vr but leaves "
                "wind_convention unset, so the vt/vr <-> u/v rotation cannot be "
                "inverted. The dataset ships vt/vr precomputed and this project "
                "never produced them, so the convention has to be measured rather "
                "than assumed - guessing it yields a wind field that is mirrored or "
                "rotated, with no error anywhere. Determine it with\n"
                "    python tools/verify_vtvr_convention.py <a *_combined.nc file>\n"
                f"on the cluster and set wind_convention in {self.model_path}.\n"
                f"Valid values: {sorted(WIND_CONVENTIONS)}"
            )
        self.model = self.load_model()
        self._check_onnx_shape()
        self.load_statistics()
        print(f"Model and weights are loaded. "
              f"polar grid (Z,R,Theta)={self.polar_shape}, "
              f"r_max={self.r_max_px:g} px ({self.r_degree_max:g} deg), "
              f"cartesian {self.cartesian_n}x{self.cartesian_n}")

    def _check_onnx_shape(self):
        """Cross-check the yaml-declared polar grid against what the onnx expects.

        Without it, a mismatch surfaces only at the first model.run() with an
        unreadable message; and if the shapes happen to be compatible while the
        geometry differs (right R, wrong r_degree_max) nothing ever complains and
        the output is quietly wrong.
        """
        want = {'input_upper': (self.polar_shape[0], self.polar_R, self.polar_theta,
                                len(self.upper_variables)),
                'input_surface': (self.polar_R, self.polar_theta,
                                  len(self.surface_variables)
                                  + (2 if self.ingest_space_info else 0))}
        for inp in self.model.get_inputs():
            if inp.name not in want:
                continue
            # Axis 0 is batch and may be dynamic (a string); compare the rest.
            got = tuple(d for d in inp.shape[1:] if isinstance(d, int))
            exp = want[inp.name]
            if len(got) == len(exp) and got != exp:
                raise ValueError(
                    f"{inp.name} shape mismatch: onnx expects {got}, the yaml "
                    f"polar block derives {exp}. Check that data_spatial_shape "
                    f"and the variable lists in {self.model_path} match the "
                    f"training config.yaml.")
        
    def normalize(self, upper_data, surface_data, reverse=False):
        if reverse:
            new_upper = upper_data * self.upper_std + self.upper_mean
            new_surface = surface_data * self.surface_std + self.surface_mean
        else:
            new_upper = (upper_data - self.upper_mean) / self.upper_std 
            new_surface = (surface_data - self.surface_mean) / self.surface_std
        return new_upper, new_surface
        
    def predict_one_step(self, input_upper, input_surface):
        
        # Validate before converting. The Cartesian domain size is derived from
        # r_degree_max / original_resolution; a mismatch means the IC was cropped
        # differently than during training, and continuing only yields wrong output.
        n = self.cartesian_n
        if input_surface.shape[:2] != (n, n) or input_upper.shape[1:3] != (n, n):
            raise ValueError(
                f"Expected a {n}x{n} Cartesian grid (derived from "
                f"r_degree_max={self.r_degree_max:g} deg / "
                f"original_resolution={self.original_resolution:g} deg), got "
                f"upper {input_upper.shape}, surface {input_surface.shape}.")

        polar_kw = dict(R=self.polar_R, Theta=self.polar_theta,
                        r_max=self.r_max_px, center_xy=self.center_xy)
        input_upper, _, theta_deg = latlon_to_polar(input_upper, **polar_kw)
        input_surface, _, _ = latlon_to_polar(input_surface, **polar_kw)

        # S4: callers speak u/v; the model wants vt/vr. Rotate here, in polar
        # space, where theta is exact per column.
        theta_rad = np.deg2rad(theta_deg)
        if self.wind_representation == 'vt_vr':
            self._rotate_wind(input_upper, input_surface, theta_rad, inverse=False)

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

        # S4: back to u/v before leaving polar space, so that what gets
        # interpolated below is a genuine vector field rather than two
        # theta-dependent scalars.
        if self.wind_representation == 'vt_vr':
            self._rotate_wind(output_upper, output_surface, theta_rad, inverse=True)

        # The four corners outside the disc (23.4% of the grid) have no model
        # output and must be NaN, not 0. Filling 0 caused B1->B2: the lon channel
        # was diluted to 0.766x, lonlat_uniformizer read a 130E domain as 99.6E,
        # and the coupling pasted the LLAT field 3,300 km too far west.
        # NaN does not spread: latlon_to_polar samples only r <= r_max so it never
        # reads the corners, LLAT->FCNV2 feedback reads only r < 7.5 deg, and the
        # FCNV2->LLAT boundary replacement overwrites the corners anyway.
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
            # B6: this line used to sit outside the if, referencing lon/lat
            # unconditionally and raising NameError when the flag was off.
            (output_surface[:, :, -2], output_surface[:, :, -1]) = np.meshgrid(lon, lat)

        return output_upper, output_surface
    
    def changing_additional_information(self, input_upper, input_surface, timestep):
        # External names throughout. By this point predict_one_step has already
        # rotated the wind back to u/v, and calc_additional_vars looks channels
        # up as ds.u / ds.v, so passing the model's own vt/vr names raises
        # AttributeError: 'Dataset' object has no attribute 'u'. Positions are
        # identical between the two lists, so the index lookups below are
        # unaffected; only the labels differ.
        additionals = recalc_additional_np(
                input_upper, input_surface, timestep,
                self.upper_variables_external, self.surface_variables_external,
                self.upper_units, self.surface_units
            )

        for v in self.surface_variables_external:
            if v in additionals:
                i = self.surface_variables_external.index(v)
                input_surface[:, :, i] = additionals[v]

        for v in self.upper_variables_external:
            if v in additionals:
                i = self.upper_variables_external.index(v)
                input_upper[:, :, :, i] = additionals[v]

        return input_upper, input_surface
    
    def IC_from_xarray_to_npy(self, IC_dataset:xr.Dataset, additional_vars=False):
        # External names: predict_one_step takes and returns u/v, so read u/v out
        # of the IC even when the model itself is trained on vt/vr. For a uv model
        # the two lists are identical.
        if not additional_vars:
            print('It needs to calc additional vars.')
            IC_dataset = calc_additional_vars(IC_dataset, True)
        input_upper = np.stack([IC_dataset[var].values for var in self.upper_variables_external], axis=-1).squeeze()
        input_surface = np.stack([IC_dataset[var].values for var in self.surface_variables_external], axis=-1).squeeze()
        lon, lat = np.meshgrid(IC_dataset.longitude, IC_dataset.latitude)
        input_surface = np.concatenate((input_surface, np.stack([lon, lat],axis=-1)), axis=-1)
        return input_upper, input_surface

    def data_to_xarray(self, upper_data, surface_data, timestep):
        DLAMPty_xr = to_xarray(upper_data, surface_data, self.upper_variables_external, self.surface_variables_external, self.upper_units, self.surface_units, self.pressure_levels)
        DLAMPty_xr = DLAMPty_xr.expand_dims(time=[pd.to_datetime(timestep)])
        return DLAMPty_xr
