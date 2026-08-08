# LLAT.ty — Polar-Coordinate Variant

A tropical-cyclone (TC) forecasting model built on a **TC-following Lagrangian limited-area
transformer**, reformulated from a Cartesian (lat, lon) grid onto **TC-centred polar
coordinates (r, θ)** so that resolution concentrates on the storm core.

The backbone is Pangu-Weather's 3-D Earth-Specific Transformer (3DEST) — a 3-D Swin
transformer in a U-Net arrangement — trained on TC-centred ERA5 reanalysis to make
autoregressive 3-hourly forecasts on a domain that moves with the storm.

> **Upstream**: derived from the `DLAMPty_polar` prototype by Y.-Y. Cheng (NTU).
> Restructured here into a flat, reproducible layout for the NCHC H200 cluster,
> with a test suite, corrected training configuration, a documented defect list,
> and a working inference and plotting path.
>
> **Reference paper**: *A Data-Driven Tropical Cyclone Model Boosted by Lagrangian
> Limited-Area Transformers* (JAS-D-26-0056).

---

## Why polar coordinates?

A TC is approximately axisymmetric about its centre. A polar grid:

- **aligns the grid with the physics** — tangential/radial structure maps onto grid axes;
- **concentrates resolution in the inner core**, where intensity and eyewall dynamics live
  (the reference paper reports systematic intensity under-prediction for severe typhoons);
- makes the coupling feedback region (a disk of fixed radius) the **natural domain shape**.

The trade-off is that a polar grid over-samples the centre and under-samples the outer
radii, and introduces a coordinate singularity at r = 0. Both are quantified below.

---

## Results

Two full training runs on 8×H200, same data and grid, differing in precision, peak
learning rate and schedule length:

| | steps | wall-clock | grad median | clipped | best val loss |
|---|---|---|---|---|---|
| **bf16 + 5e-5** | 104,999 | **6.7 h** | **0.257** | **0.0 %** | **0.24997** |
| fp32 + 4e-4 | 95,999 | 9.1 h | 0.455 | 2.6 % | 0.26871 |
| earlier bf16 + 4e-4 | 104,999 | 6.7 h | 3.8 × 10⁶ | 95.6 % | 0.30065 |

The first row is the current model. Relative to the earlier run — same variables, same
grid, so directly comparable — validation loss improves **17 %**.

Two results worth stating plainly, because both contradict what was expected:

- **Low precision was not the problem.** The failing run used bf16 and it was tempting to
  blame the format. The fp32 run at the original learning rate still spiked, clipped 2.6 %
  of its steps and reached a *worse* loss; bf16 at a lower learning rate never touched the
  clip threshold in 105,000 steps. The learning rate was the cause and precision only
  changed how badly it showed.
- **The high learning rate bought about 2,000 steps.** It led on validation loss up to
  step ~1,000 and never led again.

Throughput improved ≈ 80× over the inherited baseline, decomposable and verified:
8 GPUs (DDP was not previously engaging) × 2 (mixed precision) × 4.3 (grid) × 1.18
(removing B7) ≈ 81.

---

## Repository layout

```
.
├── train.py                     # LightningCLI entry point
├── config.yaml                  # single source of truth for training
├── run_coupled_forecast.py      # forecast driver: standalone / one-way / two-way
├── global_model/FCNV2/          # vendored FourCastNet v2 (NVIDIA, Apache 2.0)
├── interaction_tools/           # vendored FCNV2 <-> LLAT exchange
├── DLAMPty_inference.py         # inference wrapper (Cartesian ⇄ polar, vt/vr ⇄ u/v)
├── export_onnx.py               # checkpoint → ONNX, with cross-checks
├── onnx/
│   ├── LLAT_polar_vtvr_v1.yaml  # model card for the vt/vr model (this project's)
│   └── LLAT_polar_v1.yaml       # model card for the inherited u/v baseline
├── models/
│   ├── pangu_polar.py           # PanguPolarModel: circular θ padding, dual window sizes
│   ├── lightning_modules.py     # loss assembly, optimiser, scheduler
│   └── loss.py                  # WeightedL1Loss
├── utils/
│   ├── datasets.py              # ERA5TCDataset — on-the-fly Cartesian → polar resampling
│   ├── data_processor.py        # latlon_to_polar, derived variables
│   └── data_modules.py          # Lightning DataModule
├── tools/
│   ├── plot_forecast.py         # side-by-side forecast comparison figures
│   ├── verify_vtvr_convention.py# measures the dataset's vt/vr sign convention
│   ├── events_to_csv.py         # TensorBoard events → CSV (no tensorboard dependency)
│   └── compute_vtvr_stats.py    # normalisation statistics for vt/vr
├── tests/                       # pytest, CPU-only, seconds
├── experiments/                 # config overlays — one file per experiment, deltas only
├── analysis/                    # per-run reports, notebook and extracted curves
├── env_building/
│   ├── conda_env.yaml           # training environment
│   └── inference_env.yaml       # inference only (no lightning/timm/xESMF)
└── job_scripts/
    ├── calibrate.sh             # short throughput calibration (dev partition)
    └── train_h200.sh            # 8×H200, 48 h, auto-resume from last.ckpt
```

`onnx/*.onnx` is gitignored (~106 MB); copy it in separately.

---

## Training

```bash
git clone <repo-url> LLAT_polar && cd LLAT_polar
mkdir -p job_logs

# 1. Fill in your allocation and e-mail in job_scripts/*.sh
sacctmgr show assoc user=$USER format=Account,Partition -n | sort -u

# 2. Calibrate throughput (~15 min on the 4 h dev partition)
sbatch job_scripts/calibrate.sh

# 3. Set max_steps in config.yaml from the calibration — this matters, see below

# 4. Train. Chained jobs, each 48 h; the script auto-resumes from last.ckpt.
J1=$(sbatch --parsable job_scripts/train_h200.sh)
J2=$(sbatch --parsable --dependency=afterany:$J1 job_scripts/train_h200.sh)
```

### Running experiments without touching the baseline

`config.yaml` is the baseline and should not be edited per experiment. LightningCLI
merges multiple `--config` files, later overriding earlier, so each experiment is a
small overlay under `experiments/` that records only its deltas. The fully resolved
configuration is written to `<rundir>/lightning_logs/version_*/config.yaml`, so every
run is self-documenting and the baseline stays untouched.

`job_scripts/train_h200.sh` takes three environment variables so that no experiment
ever requires editing the script itself — an edited script leaves uncommitted local
changes on the cluster and blocks the next `git pull`:

| Variable  | Default | Purpose |
|-----------|---------|---------|
| `OVERLAY` | *(none)* | Extra `--config` layered on top of `config.yaml`. |
| `RUNDIR`  | `.`      | Where `lightning_logs/` and checkpoints are written. |
| `FRESH`   | `0`      | `1` ignores any existing `last.ckpt` and starts from scratch. |

```bash
sbatch --export=ALL,RUNDIR=runs/lr5e-5,OVERLAY=experiments/lr5e-5.yaml \
       job_scripts/train_h200.sh
```

Every run echoes `RunDir`, the resolved `CONDA_PREFIX`, the overlay and whether it
resumed or started fresh at the top of `job_logs/job-<id>.out`. Check those lines
before letting a job burn allocation.

Independent experiments can run concurrently — each job requests its own node, so two
jobs cost the same total GPU-hours as running them back to back but finish in half the
wall time. **Give each a distinct `RUNDIR`**: Lightning picks its `version_N` directory
by scanning for the highest existing number, so two jobs sharing a run directory race
and can collide. Reusing one `RUNDIR` across different overlays is the one case where
omitting `FRESH=1` silently corrupts the comparison, because the second job resumes the
first job's checkpoint.

### The single most important setting: `max_steps`

`max_steps` does not only bound training length — it **defines the cosine learning-rate
schedule**. Setting it far beyond what the compute budget allows means the learning rate
never anneals, and the model never reaches the regime where most of the final convergence
gain appears.

The inherited run set it to 1,600,000 and completed 252,160 steps (15.8 %), so its
learning rate decayed by **5 %** — from 4.0e-4 to 3.8e-4. It was stopped by its
wall-clock limit with validation loss still falling; the apparent plateau was
high-learning-rate oscillation, not convergence.

```
max_steps = (steps per second from calibration) × (total budget in seconds) × 0.9
```

Decide the budget *first*, then derive `max_steps`.

### What calibration tells you

| Observation | Interpretation |
|---|---|
| `it/s` on the progress bar | steps per second → derive `max_steps` |
| steps per epoch | should be ≈ `n_samples / (batch_size × n_gpu)`. If it is ≈ `n_samples / batch_size`, **DDP is not engaging** |
| GPU utilisation (`nvidia-smi`) | below ~80 % means the dataloader is the bottleneck — raise `n_workers` |

---

## Inference

```bash
conda env create -f env_building/inference_env.yaml -p ~/envs/llat_infer
conda activate ~/envs/llat_infer

# The dependency chain opens utils/land.nc and imports metpy, pysolar and
# xarray_regrid at import time; this one line exercises all of it.
python -c "from DLAMPty_inference import DLAMPty_model; print('ok')"
```

Export a checkpoint, cross-checked against the model card and verified against PyTorch:

```bash
python export_onnx.py -f <checkpoint.ckpt> \
    -o onnx/LLAT_polar_vtvr_v1.onnx \
    --yaml onnx/LLAT_polar_vtvr_v1.yaml
```

Then run a forecast. Three modes:

| Mode | Needs | Boundary |
|---|---|---|
| `standalone` | this repo + the `.onnx` | frozen at the initial condition |
| `one-way` | FCNV2 weights + the coupling repo | supplied by FCNV2 |
| `two-way` | as above | as above, plus LLAT's core is written back |

```bash
python run_coupled_forecast.py \
    --tc-id 202421W \
    --data-root /path/to/FCNV2_TC \
    --track-csv /path/to/TC_list_JMA_v2 \
    --out ~/LLAT_polar_runs \
    --mode standalone --max-starts 1 --hours 24
```

**Start with `standalone`.** It needs no global model, no GPU and no second checkout,
yet it exercises the entire polar chain — resampling, wind rotation, centre estimation,
derived-variable recomputation — so it is the smallest thing that either produces a
forecast or tells you what is broken. The forecast degrades from the edge inwards, so it
is a validation tool, not a scientific result.

`one-way` and `two-way` additionally need `--fcnv2-weight`. FCNV2 itself is vendored under
`global_model/FCNV2/` — NVIDIA's FourCastNet v2, Apache 2.0, which permits redistribution
provided the licence travels with it (`LICENSE_FourCastNetv2`). The exchange helper is
vendored under `interaction_tools/`.

That copy is deliberate but not ideal: the exchange helper is shared with the Cartesian
workflow upstream, so a fix made there will not arrive here. It is 93 KB and rarely
touched, and the alternative — depending on an unpushed branch of another repository —
proved worse in practice. FCNV2 is an SFNO, so the coupled modes also need
`torch_harmonics`; standalone imports none of it.

### Model cards

`onnx/*.yaml` describes a trained model: variable lists, grid, normalisation statistics,
weights. The `polar:` block holds **only the three values that also exist in the training
`config.yaml`** — `data_spatial_shape`, `r_degree_max`, `original_resolution`. Everything
else (radius in cells, Cartesian domain size, sampling centre, uniformiser spacing) is
derived, so changing grids is a three-line edit and a half-done change is not expressible.

Loading cross-checks the card against the ONNX input shape and rejects a mismatch, rather
than failing deep inside onnxruntime or — worse — running with a geometry that differs
from training while the shapes happen to fit.

### Wind representation

The model consumes tangential/radial wind; FCNV2, the saved output and the plotting code
all use u/v. The wrapper therefore rotates on the way in and back on the way out, **in
polar space**, where θ is exact per column. Rotating after the return to Cartesian would
mean interpolating vt/vr, whose meaning depends on θ, across neighbouring azimuths.

The sign convention is **not** hardcoded. The dataset ships vt/vr precomputed, and there
are three independent places to get a sign wrong — whether the row index runs
north-to-south, whether vt is positive counter-clockwise, whether vr is positive outward.
Guessing produces a mirrored or rotated wind field with no error raised anywhere. So it is
declared per model in the card and **measured**:

```bash
python tools/verify_vtvr_convention.py <a *_combined.nc file>
```

The tool tries all eight orthogonal candidates against a file holding u, v, vt and vr for
the same points and reports every RMSE, not just the winner: if the best is not orders of
magnitude better than the rest, the assumed geometry is wrong and the answer should not be
used. For this dataset the fit is exact (0.0000 m s⁻¹ against 10.7 for the runner-up,
agreeing between surface and upper air), giving `ccw_inward_flip`:

```
vt =  u·sin θ + v·cos θ        (positive counter-clockwise)
vr = -u·cos θ + v·sin θ        (positive INWARD)
```

That independently explains the normalisation statistics, where domain-mean vr is positive
at low levels and negative aloft — impossible under an outward-positive convention, but a
textbook secondary circulation under an inward-positive one.

### Plotting

```bash
python tools/plot_forecast.py \
    --run "LLAT polar=~/LLAT_polar_runs/.../start_from_2024102500" \
    --era5 /path/to/ERA5/for_DLAMPty \
    --tc-id 202421W --init 2024102500 --lead 24 \
    --out fig_024h.png
```

`--run` repeats to add columns; `--era5` prepends a truth column; omitting `--lead` lists
the forecast hours present. Five rows: 10 m wind with MSLP, precipitation, 850 hPa
vorticity, 700 hPa ω, 500 hPa wind with TCWV.

Three choices came out of the outer-ring investigation and are worth keeping if the script
is ever rewritten:

- **`pcolormesh`, never `contourf`.** `contourf` interpolates between real and fill values
  and invents smooth structure across the edge of the polar disc.
- **NaN draws as blank.** Outside the disc there is no model output; drawing it as 0 is
  what made earlier figures appear to have a bad outer ring.
- **`--mask-radius` defaults to 0**, showing everything. Trimming the outermost ring makes
  for a cleaner figure but hides the least constrained part of the grid, so it is opt-in.

Coastlines come from the `landmask` channel every forecast carries, recomputed from the
current lon/lat at each step, so there is no coastline file and the outline cannot drift
with the forecast. It is a 0.25° binary mask: stair-stepped, and islands below one cell
are lost. For an ERA5 column, where the derived channels are absent, the land mask is
recovered from `sst`, which ERA5 leaves undefined over land.

Each run directory carries a `run_meta.yaml` recording the channel order, so nothing
downstream has to assume a layout — the arrays are bare `.npy`, and a model-card change
would otherwise shift every field by one with nothing looking wrong.

---

## Data

```
/work/yungyun0721/TC_dataset/DLDA_data/ERA5_DLDA_data/labeled_and_obs_data_with_vt_vr
```

| | |
|---|---|
| Coverage | 2007–2020 (14 years), 394 TCs, 19,984 `*combined.nc` files |
| Training pairs | ≈ 19,590 total; 14,528 for the 2007–2017 training split |
| Source grid | 81 × 81 at **0.25°** (±10° box), 13 pressure levels |
| Model grid | **41 × 180** in (r, θ): Δr = 0.25°, Δθ = 2° |
| Wind | `vt`, `vr`, `vt10`, `vr10` (tangential/radial, NaN-free), also `u`, `v` |

Δr is matched to the source resolution: the inherited 201 × 180 grid used Δr = 0.05°,
five times finer than the data it resamples, which costs 5.5× the Cartesian grid in tokens
while adding no information.

**Resampling is on the fly** inside `ERA5TCDataset._stack_nc`, so the grid changes from
`config.yaml` alone (`data_spatial_shape` and `r_degree_max`, which must stay consistent
between the `data` and `model` sections).

---

## Configuration

| Setting | Value | Rationale |
|---|---|---|
| Grid | 41 × 180 (r, θ) | r_max = 10°, Δr matched to the 0.25° source |
| Patch / windows | (2, 8, 6) / (2, 10, 15) and (2, 8, 10) | inherited; **see the open issue below** |
| Parameters | 24,570,650 | |
| Precision | `bf16-mixed` | at LR 5e-5 the gradient norm never reached the clip threshold |
| Peak LR | 5e-5 | 4e-4 spiked in both bf16 and fp32 |
| `max_steps` | 105,000 | ≈ 9 h at the calibrated 3.6 it/s |
| Batch size | 4 per GPU (32 effective on 8 GPUs) | |
| Dataloader workers | 10 | 12 cores per GPU available; polar resampling is CPU-bound |

---

## Measured properties and known defects

Verified empirically — synthetic-field probes, direct function calls, or statistics over
saved model output — rather than by code reading alone.

### Verified non-issues

- **θ seam continuity.** Circular vs zero padding, and retaining vs removing the
  cross-seam attention mask, all yield a seam-to-interior jump ratio of ≈ 1.0 (no seam).
  Non-overlapping patch embedding means this transformer has no seam mechanism to begin
  with — unlike CNN-based polar models, which require replicated padding.
- **Strict θ-roll equivariance.** Swin window attention is equivariant only to whole-window
  rolls, at granularity `patch_θ × window_θ` (and only if `window_θ` is even and the coarse
  stage is a single window). Under the current windows no non-trivial equivariant roll
  exists. Equivariance is a diagnostic, not an acceptance criterion; the test is `xfail`.
- **Train/inference grid geometry.** The two implementations differ (`grid_sample` vs
  `map_coordinates`) but the grids match exactly: same `r`, same `θ`, same centre, same
  outside-the-disc handling.

### Open issues

| ID | Location | Issue |
|---|---|---|
| **P1** | `config.yaml` | Patch and windows are inherited from the 201-point radial grid. At R = 41 the radial axis holds 6 tokens inside a 10-wide window, 3 inside an 8-wide window after downsampling: **75 % of coarse-stage attention positions are padding**, and 6 of the 8 blocks live there. Worse for the stated goal, `patch_r = 8` at Δr = 0.25° spans 2° per patch while the radius of maximum wind is 0.3–0.5°, so **the entire inner core falls inside one patch**. Two alternatives measured: R = 40 with patch (2,4,6) and uniform (2,6,15) windows costs 0.68× the attention and raises the real-token share to 83 %; the CLAUDE.md target grid (40 × 96, patch (2,4,4)) costs 0.44× — but Θ = 96 loses half the peak intensity over 80 autoregressive steps unless the state stays in polar throughout, so the two changes are coupled. |
| **B3** | `DLAMPty_inference.py`, `utils/datasets.py` | `r = linspace(0, r_max, R)` starts at zero, so the innermost ring samples one physical point 180 times. `r_min = Δr/2` fixes it, but **training and inference must change together**. |
| **B5** | `utils/datasets.py` | Operator precedence: `int((x - int(...)+1)/2)` evaluates as `(x-79)/2` rather than `(x-81)/2` — an off-by-two crop. Only triggers when the source grid is not already 81 × 81. |

### Fixed

| ID | Was |
|---|---|
| **B1** | `polar_to_latlon(fill_value=0.0)` wrote physical zeros outside r_max — 1,536 cells, 23.4 % of the frame. Now NaN. |
| **B2** | The TC centre was the plain mean over the whole array. Fed B1's zeros it read a 130°E domain as 99.6°E, **3,300 km** out, and compounded: the corrupted axis is written back into the state, so the centre collapsed by a further 0.766× each step. Now nan-aware throughout — including the two means whose only use is their *sign*, which decide axis direction and would otherwise silently reverse the longitude axis. |
| **B4** | `export_onnx.py` used a hardcoded Cartesian input shape. Now derived from the checkpoint's own hyperparameters, cross-checked against the model card, and the exported graph is compared against PyTorch. |
| **B6** | A `np.meshgrid` assignment was indented outside its `if`, raising `NameError` whenever `uniformize_lonlat` was false. |
| **B7** | `earth_specific_bias` was an all-zero plain tensor — not a `Parameter`, not a buffer — so adding it was a no-op, yet each of 16 blocks indexed a ~26 MB tensor **on CPU** and copied it to GPU every forward. Removed; **bit-identical** output, **18 % faster** on CPU. |

### Outer-ring artefact

Saved forecasts contained 176 unphysical cells (2.7 %), all at r = 39.3–40.0 px — the
outermost ~0.2° — with mean sea-level pressure as low as 423 hPa. Both coordinate
transforms were verified clean against uniform-field probes, so the artefact originates in
the model output at large radius. The leading hypothesis is that an **unweighted loss
under-constrains the outer domain**: every ring carries the same number of grid points but
ring area grows with r, so per-unit-area weight scales as 1/r — an ≈ 80× bias toward the
innermost ring. The core (r < 8°) is unaffected, and the coupling overwrites r > 8° each
step, so it does not accumulate — but note that only applies to the *replaced* variables,
and only every 6 h, while LLAT steps 3 h.

---

## Improvement roadmap

| Priority | Change | Expected effect |
|---|---|---|
| 1 | Redesign patch and windows for R = 41 (P1) | more radial resolution *and* less compute; the current setting also defeats the inner-core goal |
| 2 | Add a configurable radial loss weight `w(r) = r^p` | targets the outer-ring artefact; `p = 1` is the equal-area baseline |
| 3 | Keep the autoregressive state in polar throughout inference | removes 2 interpolations per step, ≈ 160 over a 10-day forecast; also unlocks a coarser Θ |
| 4 | `r_min = Δr/2` (B3) | removes the coordinate singularity; matters most for vt/vr |
| 5 | Add a magnitude-weighted loss term | addresses the intensity under-prediction reported in the paper |
| 6 | `torch.compile`, activation checkpointing | 10–50 % and larger batches respectively |

---

## Testing

```bash
python -m pytest tests/ -q          # 85 passed, 3 xfailed, CPU only, ~1 min
```

| File | Covers |
|---|---|
| `test_theta_equivariance.py` | forward pass, shapes, determinism; θ-roll diagnostic (`xfail`) |
| `test_trainability.py` | every parameter receives a finite gradient; pins B7 |
| `test_polar_roundtrip.py` | B1/B2/B6 — NaN outside the disc, centre survives the round trip, axis direction, and a negative control reproducing the old 0.766× collapse |
| `test_inference_config.py` | grid derivation from the model card, and that `predict_one_step` actually uses it |
| `test_wind_rotation.py` | the rotation is orthogonal, invertible and speed-preserving for all eight conventions; the two rotations cancel end to end |
| `test_driver.py` | derived values reproduce the literals they replaced; standalone's frozen ring |
| `test_plot.py` | vorticity against solid-body rotation; channel order matches the model card |

These are architecture- and wiring-level properties: they hold for any weights, so they run
in seconds on CPU with random tensors and catch structural defects before any GPU time is
spent.

Several tests carry a **negative control** — deliberately restoring the defect and
asserting the test fails. Without one, a test cannot show it has any power. Two examples
that mattered: asserting derived attributes passed even when `predict_one_step` ignored
them, and a wind-rotation test built on white noise passed regardless, because the two
bilinear interpolations in the path destroy noise and swamp the effect being measured.
