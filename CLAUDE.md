# CLAUDE.md

Working notes for agents. The README describes what the project *is*; this file
records what is **not derivable from the code** — the invariants that look like
details but are load-bearing, and the traps that have already cost time.

Read the "Invariants" section before changing anything under `DLAMPty_inference.py`,
`run_coupled_forecast.py`, or `utils/data_processor.py`.

---

## Current status — 2026-08-11

Keep this block short and current. It is the first thing an agent reads, so it should
answer "where is this project" in ten lines, not narrate history. Detail belongs in dated
notes under `analysis/`; what was *done* belongs in the git log.

**Works.** Training (val loss 0.24997, bf16 + LR 5e-5). ONNX export with verification.
All three forecast modes to +240 h — a 10-day one-way forecast takes about a minute, so
whole-season experiments are cheap. Five diagnostics: comparison figure, radial profile,
track error with map, steering flow, vortex centre.

**The track error is diagnosed.** See `analysis/2026-08-11_track_error_diagnosed.md`.
`val_RMSE/lon` is 0.774° and `val_RMSE/lat` 0.554°, a **102 km position error for one 3 h
step** against the 104 km the storm covers in it; as a random walk that is 288 km at 24 h
and the observed error was 296 km. The model's own steering flow is right (identical at
hour 0 by construction) but its frame moves at **55–72 %** of what that flow implies, and
the vortex does not drift to compensate (median 30 km from the array centre over 240 h).
The two channels that carry the whole track are **0.43 %** of the objective.

**Blocked on nothing.**

**Next, in order.** (1) `--frame-speed-scale ≈ 1.45` at inference — no retraining, measures
how much of the error is pure speed calibration. (2) persistence baseline, without which
"102 km per step" cannot be called worse than doing nothing. (3) `residual: true`, masked
to lon/lat, and `surface_var_weights` — both need a retrain. (4) P1, which is now about
structure and intensity, not track.

**Dead hypotheses, do not revisit without new evidence.** The lateral boundary (one-way and
standalone agree to +96 h). The environment representation, i.e. P1 as a *track* problem
(the steering flow is right). Vortex drift (there is none). The outer-ring artefact (truth
shows the same rise). Best-track provenance (inference is JMA, training was JTWC per the
paper, but vortex-to-vortex agrees with frame-to-frame, bounding it at ~30 km).

**202421W at 2024-10-25 00Z is a 35 kt, 998 hPa storm.** `wind_min` and `vort850` both
mislocate it at hour 0, on an ERA5 analysis. Use `mslp`. The paper's Fig. 4b uses the same
storm 12 h later, so it is not a like-for-like comparison, and the paper reports ~30 %
larger track errors for weak samples.

**Recently learned, worth not relearning.** NaN outside the polar disc is not inert: the
rim of `latlon_to_polar` interpolates bilinearly and reaches across it, so 23.4 % NaN
corners become 2.9 % of the polar array, and one NaN makes the entire model output NaN
because attention mixes every token. Corners must be refilled after *every* step. The
symptom appeared two stages downstream, in `lonlat_uniformizer` and then `xarray_regrid`,
and the first diagnosis from that traceback was wrong — the guard now in
`predict_one_step`, which names the offending channels, is what corrected it.

---

## What this is, in three lines

TC forecasting model: Pangu-Weather 3DEST (3-D Swin transformer, U-Net arrangement) on a
**TC-following Lagrangian** limited-area grid, 3-hourly autoregressive. Reformulated from
Cartesian (lat, lon) onto **TC-centred polar (r, θ)**. Trained on TC-centred ERA5;
at inference FCNV2 supplies the lateral boundary.

Current model: `onnx/LLAT_polar_vtvr_v1.yaml` — 41 × 180 (r, θ), vt/vr wind, bf16, LR 5e-5,
best val loss 0.24997.

---

## Environments

Three, and they are not interchangeable.

| Where | Name | For |
|---|---|---|
| NCHC H200 cluster | `ty` (micromamba) | training |
| Lab host `neo82` | `~/envs/llat_infer` | inference, plotting |
| Local Windows | `ty-dev` (conda, CPU torch) | tests, analysis notebooks |

`/mamba/envs` on the lab host belongs to another user and is not writable — build
environments under `~/envs` with `conda create -p`.

**Do not run the derived-variable chain locally.** `calc_additional_vars` regrids a global
land mask and evaluates solar position per grid point; on Windows it appears to hang. Test
the *wiring* instead by intercepting `recalc_additional_np` — see
`test_wind_rotation.py::test_additional_vars_recomputed_with_external_names`.

---

## Invariants

Each of these has already been broken once. None of them fail loudly.

### 1. `lon` / `lat` are the moving frame, not weather

They are the last two surface channels and they are **predicted variables**. The grid
follows the storm, so "the TC moved" is implemented as "the lon/lat field shifted", and
downstream code recovers the TC centre by **averaging the field**.

That identity — mean equals centre — holds only over a set that is symmetric about the
centre. Consequences:

- Filling part of the field with anything else breaks it. Filling the corners with 0
  diluted a 130°E domain to 99.6°E, **3,300 km** out, and compounded 0.766× per step (B2).
- Freezing part of it at the initial condition mixes two centres and the storm moves at the
  unfrozen fraction of the intended speed — 49 % at the coupling radius.
- Masking with NaN is fine: the remaining disc is still centrally symmetric, which is why
  the fix works. Use `nanmean`, including for the two means whose only use is their *sign*
  (they set axis direction; a NaN there silently reverses the longitude axis).

### 2. Internal names are vt/vr, external names are u/v

The model consumes tangential/radial wind. Everything outside the wrapper — FCNV2, the
saved `.npy`, the plotting — uses u/v. `predict_one_step` rotates both ways, in **polar
space**, where θ is exact per column.

So `DLAMPty_model` carries two lists: `upper_variables` (the model's, keyed by the
normalisation statistics) and `upper_variables_external` (u/v). Anything handing arrays to
or from `predict_one_step` needs the **external** names. Missing one of these is how the
first forecast attempt died: `AttributeError: 'Dataset' object has no attribute 'u'`.

### 3. The sign convention of vt/vr was measured, not chosen

The dataset ships vt/vr precomputed. There are three independent places to get a sign wrong
(row order, cyclonic sense, inward vs outward), and a wrong guess produces a mirrored wind
field with no error anywhere. `tools/verify_vtvr_convention.py` measured it against a real
file: `ccw_inward_flip`, RMSE exactly 0.0000 against 10.7 for the runner-up.

**If the dataset is ever regenerated, re-run that tool.** Do not assume it carries over.

### 4. NaN outside the polar disc is deliberate

The disc covers 76.6 % of the square frame; the corners have no model output. They are NaN
on purpose. Do not "clean them up" — plotting them as 0 is what made earlier figures look
like the model had a bad outer ring, and averaging over them is B2.

NaN does not spread: `latlon_to_polar` samples only r ≤ r_max, LLAT→FCNV2 feedback reads
only r < 7.5°, and FCNV2→LLAT boundary replacement overwrites the corners anyway.

### 5. The grid comes from the model card, never from a literal

`onnx/*.yaml` holds the three values that also exist in the training `config.yaml`
(`data_spatial_shape`, `r_degree_max`, `original_resolution`); everything else is derived.
This was S2: the grid used to be hardcoded in `predict_one_step` as R=201/Θ=180/r_max=40.

The same rule applies to consumers of the output. Each run directory carries
`run_meta.yaml` recording the channel order, because the arrays are bare `.npy` and a card
change would otherwise shift every field by one with nothing looking wrong.

---

## Known defects

The README has the full table. The ones most likely to matter while working:

- **P1 — patch/window sized for the wrong grid.** `patch_shape: [2, 8, 6]` and the two
  window sizes were inherited from the 201-point radial grid. At R=41 three quarters of
  coarse-stage attention positions are padding, and `patch_r = 8` at Δr 0.25° spans 2° per
  patch while the radius of maximum wind is 0.3–0.5° — **the whole inner core lands in one
  patch**, defeating the point of the polar grid. Fixing it is also *cheaper*. Top of the
  roadmap.
- **B3 — `r = linspace(0, r_max, R)` starts at zero**, so the innermost ring samples one
  point 180 times. `r_min = Δr/2` fixes it, but **training and inference must change
  together** or the geometries diverge silently.
- **B5 — off-by-two crop** in `datasets._trim_var`, only triggered when the source grid is
  not already 81 × 81.

---

## Working practices

### Commits and comments in English

Everything that lands in the repo is English: commit messages, comments, docstrings, YAML
prose. **No `Co-Authored-By` trailer.** The repo is public and presented as portfolio work.
Chat and the analysis reports outside the repo stay in 繁體中文.

Pass multi-line commit messages via a file (`git commit -F`); backticks in a shell
heredoc get eaten by command substitution and have silently truncated a message here before.

### Tests need a negative control

A test that cannot fail proves nothing. Twice a test here passed while measuring nothing:

- asserting derived attributes (`m.polar_R`) passed even when `predict_one_step` ignored
  them and used a literal — fixed by intercepting what the model actually receives;
- a wind-rotation test on white noise passed for every convention, because the two bilinear
  interpolations in the path destroy noise and swamp the effect. Fixed by comparing against
  a `uv` model, so the interpolation error is common to both paths and cancels.

When adding a test for a fix, restore the defect and confirm the test goes red.

### The cluster job script is driven by environment variables

`OVERLAY`, `RUNDIR`, `FRESH` — never edit `job_scripts/train_h200.sh` for an experiment. An
edited script leaves uncommitted changes on the cluster and blocks the next `git pull`,
which has already happened. Parallel experiments need distinct `RUNDIR`; reusing one across
overlays silently resumes the wrong checkpoint.

### Start inference with `--mode standalone`

It needs only this repo plus the `.onnx` — no FCNV2 weights, no GPU, no second checkout —
yet exercises the entire polar chain. It is a validation tool, not a scientific result: the
boundary is frozen at t=0, so the forecast degrades from the edge inwards and the track is
not meaningful.

---

## Relationship to `couple_FCNV2_LLAT`

That repository is where FCNV2 and the exchange helper came from; both are now **vendored**
here under `global_model/FCNV2/` and `interaction_tools/`, so this repo runs on its own.
The trigger was practical: the polar branch of the exchange helper was never pushed and
existed only as an uncommitted working-tree change, so depending on that checkout meant
depending on something that did not exist remotely.

The cost is real — the exchange helper is shared with the Cartesian workflow upstream, so a
fix made there will not arrive here. If it is ever edited upstream, diff the two.
`POLAR_ANALYSIS_REPORT.md` there remains the long-form investigation record.

The exchange looks channels up by name (`.index('u10')`), which is why the driver hands it
the **external** names — see invariant 2.

---

## Paths

| What | Where |
|---|---|
| Training data | `/work/yungyun0721/TC_dataset/DLDA_data/ERA5_DLDA_data/labeled_and_obs_data_with_vt_vr` (NCHC) |
| FCNV2 weights | `/wk2/yungyun/code_space/FCNV2_test/weight` (lab host) |
| Forecast IC | `/wk2/yungyun/FCNV2_TC/{TC_ID}/ERA5/for_{FCNV2,DLAMPty}` (lab host) |
| Best-track CSV | `/wk2/yungyun/ERA5_2024_for_TC/TC_list_JMA_v2` (lab host) |

`onnx/*.onnx` is gitignored (~106 MB) and must be copied separately.
