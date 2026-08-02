# LLAT.ty — Polar-Coordinate Variant

A tropical-cyclone (TC) forecasting model built on a **TC-following Lagrangian limited-area
transformer**, reformulated from a Cartesian (lat, lon) grid onto **TC-centred polar
coordinates (r, θ)** so that resolution concentrates on the storm core.

The backbone is Pangu-Weather's 3-D Earth-Specific Transformer (3DEST) — a 3-D Swin
transformer in a U-Net arrangement — trained on TC-centred ERA5 reanalysis to make
autoregressive 3-hourly forecasts on a domain that moves with the storm.

> **Upstream**: derived from the `DLAMPty_polar` prototype by Y.-Y. Cheng (NTU).
> Restructured here into a flat, reproducible layout for the NCHC H200 cluster,
> with a test suite, corrected training configuration, and a documented defect list.
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

## Repository layout

```
.
├── train.py                  # LightningCLI entry point
├── config.yaml               # single source of truth for training
├── DLAMPty_inference.py      # ONNX inference wrapper (Cartesian ⇄ polar internally)
├── export_onnx.py            # checkpoint → ONNX
├── models/
│   ├── pangu_polar.py        # PanguPolarModel: circular θ padding, dual window sizes
│   ├── lightning_modules.py  # loss assembly, optimiser, scheduler
│   └── loss.py               # WeightedL1Loss
├── utils/
│   ├── datasets.py           # ERA5TCDataset — on-the-fly Cartesian → polar resampling
│   ├── data_processor.py     # latlon_to_polar, derived variables
│   └── data_modules.py       # Lightning DataModule
├── tests/                    # pytest: smoke, determinism, gradient coverage, equivariance
└── job_scripts/
    ├── calibrate.sh          # short throughput calibration (dev partition)
    └── train_h200.sh         # 8×H200, 48 h, auto-resume from last.ckpt
```

---

## Quick start (NCHC H200 cluster)

```bash
git clone <repo-url> LLAT_polar && cd LLAT_polar
mkdir -p job_logs

# 1. Fill in your allocation and e-mail in job_scripts/*.sh
sacctmgr show assoc user=$USER format=Account,Partition -n | sort -u

# 2. Calibrate throughput (~15 min on the 4 h dev partition)
sbatch job_scripts/calibrate.sh

# 3. Set max_steps in config.yaml from the calibration (see below) — this matters

# 4. Train. Chained jobs, each 48 h; the script auto-resumes from last.ckpt.
J1=$(sbatch --parsable job_scripts/train_h200.sh)
J2=$(sbatch --parsable --dependency=afterany:$J1 job_scripts/train_h200.sh)
```

Run the test suite with `python -m pytest tests/ -v`.

---

## The single most important setting: `max_steps`

`max_steps` does not only bound training length — it **defines the cosine
learning-rate schedule**. Setting it far beyond what the compute budget allows means the
learning rate never anneals, and the model never reaches the fine-tuning regime where most
of the final convergence gain appears.

Evidence from the previous run (recovered by parsing its 196 MB TensorBoard event file):

| Quantity | Value |
|---|---|
| Wall-clock | 8.32 days |
| Steps completed | **252,160 / 1,600,000 = 15.8 %** |
| Epochs | 70 |
| Validation loss | 0.3256 → 0.2906 (best) — **still decreasing at cut-off** |
| Learning rate | 4.0e-4 → 3.8e-4 (**decayed by only 5 %**) |
| Train vs val loss | 0.2954 vs 0.2931 — **no overfitting whatsoever** |

The run was terminated by its wall-clock limit, not by convergence. The apparent plateau is
a high-learning-rate oscillation, not a converged model.

**Rule to follow instead:**

```
max_steps = (steps per second from calibration) × (total budget in seconds) × 0.9
```

Decide the budget *first*, then derive `max_steps`, so the schedule completes exactly when
the allocation runs out.

---

## What calibration tells you

| Observation | Interpretation |
|---|---|
| `it/s` on the progress bar | steps per second → derive `max_steps` |
| steps per epoch | should be ≈ `n_samples / (batch_size × n_gpu)`. If it is ≈ `n_samples / batch_size`, **DDP is not engaging** |
| GPU utilisation (`nvidia-smi`) | below ~80 % means the dataloader is the bottleneck — raise `n_workers` |

---

## Data

```
/work/yungyun0721/TC_dataset/DLDA_data/ERA5_DLDA_data/labeled_and_obs_data_with_vt_vr
```

| | |
|---|---|
| Coverage | 2007–2020 (14 years), 394 TCs, 19,984 `*combined.nc` files |
| Training pairs | ≈ 19,590 total; ≈ 15,400 for the 2007–2017 training split |
| Source grid | 81 × 81 at **0.25°** (±10° box), 13 pressure levels |
| Model grid | **201 × 180** in (r, θ): Δr = 0.05°, Δθ = 2° |
| Extra fields | `vt`, `vr`, `vt10`, `vr10` (tangential/radial wind, NaN-free), plus `ws`, `vort`, `theta_e` |

Tangential/radial winds are already present in the dataset, so a Vt/Vr ablation requires
only a variable-list change — no preprocessing.

**Resampling is performed on the fly** inside `ERA5TCDataset._stack_nc`, so the grid can be
changed from `config.yaml` alone (`data_spatial_shape` and `r_degree_max`, which must be
kept consistent between the `data` and `model` sections).

---

## Configuration

| Setting | Value | Rationale |
|---|---|---|
| Grid | 201 × 180 (r, θ) | r_max = 10° |
| Patch / windows | (2, 8, 6) / (2, 10, 15) and (2, 8, 10) | layers 1 & 4 / layers 2 & 3 |
| Parameters | 24,570,650 | |
| Precision | `16-mixed` | the reference model trained successfully in fp16; fp32 halves throughput. Use `bf16-mixed` if instability appears — not `32` |
| Batch size | 4 per GPU (32 effective on 8 GPUs) | |
| Dataloader workers | 10 | the cluster allows 12 cores per GPU; polar resampling is CPU-bound |

---

## Measured properties and known defects

All findings below were verified empirically (synthetic-field probes, direct function calls,
or statistics over saved model output) rather than by code reading alone.

### Verified non-issues

- **θ seam continuity.** Circular vs zero padding, and retaining vs removing the
  cross-seam attention mask, all yield a seam-to-interior jump ratio of ≈ 1.0 (no seam).
  Non-overlapping patch embedding means this transformer has no seam mechanism to begin
  with — unlike CNN-based polar models, which require replicated padding.
- **Strict θ-roll equivariance.** Swin window attention is equivariant only to whole-window
  rolls, at granularity `patch_θ × window_θ` (and only if `window_θ` is even and the coarse
  stage is a single window). Under the current window choice no non-trivial equivariant roll
  exists. Equivariance is therefore a diagnostic, not an acceptance criterion; the
  corresponding test is marked `xfail`.

### Open defects

| ID | Location | Issue | Affects |
|---|---|---|---|
| B1 | `DLAMPty_inference.py:319` | `polar_to_latlon(fill_value=0.0)` writes **physical zeros** outside r_max (1,536 cells, 23.4 % of the frame, in 18 of 20 channels). Should be NaN. | inference, plotting |
| B2 | `utils/data_processor.py:212` | TC centre is the plain mean over the whole array. Fed the zeros from B1, it returns 104.93°E instead of 137.00°E — a **≈ 3,400 km** offset. Latent: the archived run was unaffected, so the executed code differed from what is in the tree. | inference |
| B3 | `DLAMPty_inference.py:43` | `r = linspace(0, r_max, R)` starts at zero, so the innermost ring samples one physical point 180 times (coordinate singularity). Use `r_min = Δr/2`. | inference, data |
| B4 | `export_onnx.py` | `input_sample` still uses the 81 × 81 Cartesian shape; shape mismatch for the polar model. | export |
| B5 | `utils/datasets.py:131` | Operator precedence: `int((x - int(...)+1)/2)` evaluates as `(x-79)/2` rather than `(x-81)/2` — an off-by-two crop. Only triggers when the source grid is not already 81 × 81. | data |
| B6 | `DLAMPty_inference.py:331` | The `np.meshgrid` assignment is indented outside its `if` block; `NameError` if `uniformize_lonlat` is ever false. | inference |
| B7 | `models/pangu_polar.py:728` | `earth_specific_bias` is an all-zero plain tensor (not a `Parameter`, not a buffer). Adding it is a mathematical no-op, yet each of the 16 blocks indexes a ~26 MB tensor **on CPU** and copies it to GPU every forward pass. | **training throughput** |

B1–B6 are confined to the inference path and do not affect training. **B7 does** — deleting
the dead bias computation is numerically identical and removes roughly 400 MB of
host-to-device traffic per forward pass.

### Outer-ring artefact

Saved forecasts contain 176 unphysical cells (2.7 %), all at r = 39.3–40.0 px — the outermost
~0.2° — with mean sea-level pressure as low as 423 hPa. Both coordinate transforms were
verified clean against uniform-field probes, so the artefact originates in the model output
at large radius. The leading hypothesis is that an **unweighted loss under-constrains the
outer domain**: every ring carries the same number of grid points, but ring area grows with
r, so per-unit-area weight scales as 1/r — an ≈ 80× bias toward the innermost ring. The core
(r < 8°) is unaffected, and because the coupling overwrites r > 8° each step, the artefact
does not accumulate.

---

## Improvement roadmap

| Priority | Change | Expected effect |
|---|---|---|
| 1 | Set `max_steps` from the actual budget | lets the cosine schedule anneal — likely the single largest gain |
| 2 | `precision: 16-mixed` | ≈ 2× throughput (already applied) |
| 3 | Remove the dead bias computation (B7) | removes per-step host-to-device traffic |
| 4 | Coarsen the radial grid (Δr = 0.05° over-samples a 0.25° source) | 201 × 180 costs 5.5× the Cartesian grid; a matched grid would cut this substantially |
| 5 | Add a configurable radial loss weight `w(r) = r^p` | targets the outer-ring artefact; `p = 1` is the equal-area baseline |
| 6 | Add a magnitude-weighted loss term | addresses intensity under-prediction |
| 7 | Keep the autoregressive state in polar throughout inference | removes 2 interpolations per step (≈ 160 over a 10-day forecast) |
| 8 | `torch.compile`, activation checkpointing | 10–50 % and larger batches respectively |

---

## Testing

```bash
python -m pytest tests/ -v
```

| Test | Purpose |
|---|---|
| `test_smoke` | forward pass, output shapes, finiteness |
| `test_determinism` | no hidden stochasticity in `eval()` mode |
| `test_backward_grad_coverage` | every parameter receives a finite gradient |
| `test_dead_bias_landmine` | pins the B7 defect; flip the assertion once fixed |
| `test_theta_equivariance` | `xfail` numerical diagnostic (see above) |

These are architecture-level properties: they hold for any weights, so they run in seconds on
CPU with random tensors and catch structural defects before any GPU time is spent.
