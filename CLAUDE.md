# CLAUDE.md

Working notes for agents. The README describes what the project *is*; this file
records what is **not derivable from the code** — the invariants that look like
details but are load-bearing, and the traps that have already cost time.

Read the "Invariants" section before changing anything under `DLAMPty_inference.py`,
`run_coupled_forecast.py`, or `utils/data_processor.py`.

---

## Current status - 2026-08-17

Keep this block short and current. It is the first thing an agent reads. Detail belongs in
dated notes under `analysis/` - which is **gitignored**, so those exist only on the machine
that wrote them; anything that must survive a clone belongs here or in a commit message.

**Two models trained, same steps and schedule, geometry the only difference.**

| | grid | val loss |
|---|---|---|
| `LLAT_polar_vtvr_v1` | R=41, patch (2,8,6), windows (2,10,15)/(2,8,10) | 0.24997 |
| **`LLAT_polar_p1_v1`** | **R=40, patch (2,4,6), windows (2,10,15)/(2,5,15)** | **0.24555** |

**P1 is measured**, on the 81 initial times both runs have.

*Intensity: confirmed, decisively.* 202421W from 2024102700 reaches **931.7 hPa** against
the baseline's 969.4 and ERA5's 934.2 - the baseline was 35 hPa short of its own training
target and P1 matches it. The life cycle is still wrong (monotonic deepening, peak 54 h
late, no decay), and past +72 h the track error puts the storms in different environments,
so read intensity only where the track holds.

*Track: no net change.* 24 h 141 -> 140 km, 48 h 269 -> 242 (-10 %), 96 h 567 -> 712
(+26 %), 120 h 757 -> 746. The worst cases are different storms and P1's worst is larger.
It moved the error around.

*One prediction failed.* Spurious recurvature was predicted not to improve; on 202414W
2024091600 it went 2207 -> 772 km, and then the season showed the win did not generalise.
The reasoning that broke: `steering.py` had the 500 km areal-mean steering matching ERA5,
read as the environment being adequately represented, but **a correct mean says nothing
about the radial structure of the flow** - which 6 radial tokens, 3 after downsampling and
72 % padding, cannot hold. Seeing the steering and acting on it are different things.

**What P1 actually did to the track: it lowered the model's willingness to turn north.**
Its three worst cases are the mirror image of the three it fixed. 202414W twice and 202422W
are storms that should NOT have recurved and the baseline turned them north; P1 fixed all
three, 202422W completely (1989 -> 444 km). 202419W and 202417W are storms that SHOULD have
recurved sharply - ERA5 takes them to 48.8 N and 42.0 N - and P1 under-turns them worse than
the baseline (2517 against 1419, 2163 against 1701), with the error almost entirely
along-track. 202416W should have slowed and P1 overshot further east.

So it traded false recurvatures for missed ones. That is a shifted bias, not added skill,
and it is why the season median is a wash while individual cases move by factors of three.
Any future change should be judged this way - which direction of error it trades for which -
rather than on a median that averages the two together.

So P1 bought a large, mechanism-backed intensity gain at no net track cost, and is not the
track fix. The polar-specific spurious recurvature stays the open question: P1 changed which
storms it happens to, not whether it happens.

**Two failure modes, different causes, do not conflate them.**

*The speed deficit* is the season median. Multiplicative at **0.78x** over 1276 intervals,
worst (0.68) in the 25-30 N recurvature band. It is the coordinate channels: `val_RMSE/lon`
0.774° and `val_RMSE/lat` 0.554° is **102 km for one 3 h step** against 104 km of storm
motion, which as a random walk gives 288 km at 24 h where 296 was observed. Those two
channels are **0.43 %** of the objective. `--frame-speed-scale 1.45` beats 1.0 at all 32
leads (24 h: 296 → 99 km) but no one factor wins everywhere.

*Spurious northward recurvature* is the season tail and is **polar-specific and
unexplained**. The three worst cases all turn north while the storm goes west; the Cartesian
v57_5d on the same cases is 3-6x better and turns the right way. This is the most
interesting open question.

**Ruled out - do not revisit without new evidence.** The lateral boundary; domain size (the
steering that best predicts motion is at 4°, well inside the 10° disc); vortex drift (30 km
median over 240 h); the outer-ring artefact; best-track provenance (bounded at ~30 km);
the vt/vr convention (0.0000 RMSE across 2007/2013/2020, both pairs); a transform sign error
(all four directions survive to 0.00); an inverted meridional coupling (one case, three
refute it); and **B3**, where cell-centred rings measured *worse* than the degenerate r=0
ring, which spends 179 duplicate samples to capture the centre exactly.

**A method worth reusing.** Anything that is only a change of representation - a grid, an
interpolation, a sampling - can be tested before training, because no learning is involved.
The B3 round-trip took a second and stopped a 7-hour retrain aimed at making things worse.

**After P1, in order.** (1) If the recurvature survives: train polar on **u/v** instead of
vt/vr - the last structural difference from the Cartesian model, and a uniform flow is two
constants in Cartesian but a wavenumber-1 phase in θ in polar. (2) `residual` masked to
lon/lat, and `surface_var_weights`; with a residual connection persistence becomes the
model's floor, which it currently is not. (3) Rollout fine-tuning.

**This model is data-limited**: 14,522 samples, 24.6 M parameters, 231 epochs, a 15.2 %
train/val gap. More resolution can buy memorisation as easily as skill - which is why every
claim gets checked against a season and not a case.

---

## What this is, in three lines

TC forecasting model: Pangu-Weather 3DEST (3-D Swin transformer, U-Net arrangement) on a
**TC-following Lagrangian** limited-area grid, 3-hourly autoregressive. Reformulated from
Cartesian (lat, lon) onto **TC-centred polar (r, θ)**. Trained on TC-centred ERA5;
at inference FCNV2 supplies the lateral boundary.

Current model: `onnx/LLAT_polar_vtvr_v1.yaml` — 41 × 180 (r, θ), vt/vr wind, bf16, LR 5e-5,
best val loss 0.24997.

---

## The dataset, counted

`end_year` is exclusive in `datasets.py`, so the config's apparent 2018 overlap is not one.
A sample is one 3-hourly **pair**, not one storm: samples = files − cases.

| | years | cases | files | samples |
|---|---|---|---|---|
| train | 2007–2017 | 302 | 14,824 | **14,522** |
| val | 2018–2019 | 66 | 3,958 | 3,892 |
| test | 2020 | 26 | 1,202 | 1,176 |

**`batch_size: 4` is per GPU.** On 8 H200s the effective batch is 32, so an epoch is
14,522/32 = 454 steps — the log measures 455 — and the 105,000-step run is **231 epochs**,
not the 30 a single-GPU reading would suggest.

231 passes over 14.5k samples with **24.6 M parameters** leaves a **15.2 %** train/val gap
(0.21697 against 0.25000 at the selected step). For scale, GraphCast and FourCastNet train
on roughly 55,000 samples. **This model is data-limited, not capacity-limited**, which
matters for P1: more inner-core resolution can as easily buy more memorisation. Judge any
retrain against a whole-season baseline, never one case.

The 2024 files under `/wk2/yungyun/FCNV2_TC/` on the lab host are **not** part of any of
this. They are cut for inference by `download_ERA5_from_google_according_BT.py`, and 2024 is
outside train, val and test alike — which makes it a clean evaluation year.

---

## Environments

Three, and they are not interchangeable.

| Where | Name | For |
|---|---|---|
| NCHC H200 cluster | `ty` (micromamba) | training |
| Lab host `neo82` | `~/envs/llat_infer` | inference, plotting |
| Local Windows | `ty-dev` (conda, CPU torch) | tests, analysis notebooks |

`/mamba/envs` on the lab host belongs to another user and is not writable — build
environments under `~/envs` with `conda create -p`. `conda install` from `(base)` there fails
with `EnvironmentNotWritableError`, which reads like a permissions change and is not one.

Tools live in the environment that installed them, `git-lfs` included, so a `git clone` of an
LFS repository succeeds under `llat_infer` and the same repository's `git restore` fails under
`(base)` with `git-lfs: not found`. When only one large file is wanted, skip git entirely:
`curl -L https://media.githubusercontent.com/media/{owner}/{repo}/{ref}/{path}` fetches LFS
content directly, where the usual `.../raw/...` URL returns a 130-byte pointer.

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

Checked across the training set as well, after a forecast turned north while its own
winds pushed south and a sign error looked plausible: 2007, 2013 and 2020, **both** the
`vt10/vr10` surface pair and the `vt/vr` upper-air one, all `ccw_inward_flip` at 0.0000
RMSE. The convention is uniform over fourteen years and matches inference. Combined with
`tools/meridional_check.py` finding the transforms clean in all four cardinal directions,
**the wind representation is not where the spurious recurvature comes from** — do not spend
time here again without new evidence.

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

### Resuming was never exercised until it had to be

`train_h200.sh` resumes from `last.ckpt` whenever `FRESH` is unset, and that path
had never run: every training so far either started fresh or finished. The first
job that needed to continue died in twenty-six seconds, because LightningCLI reads
the checkpoint with `torch.load(weights_only=True)` to resolve the config, and
PyTorch 2.4+ refuses the numpy scalars a Lightning checkpoint stores as logged
metrics. `train.py` now allowlists them; the general point is that a code path
nothing has run is not a working code path, however plainly the script advertises it.

Checkpoints are also 305 MB each and `save_top_k` defaulted to 30, which is 9.2 GB
per run. Set `SAVE_TOP_K=5` on the `sbatch` line. A run that fills the quota dies
mid-write with a truncated file and an exception that says nothing about disks.

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
| Forecast output | `~/LLAT_polar_runs/{TC_ID}/{mode}_{version}/start_from_{stamp}` (lab host) |

Note which machine. Training is on NCHC, inference and every forecast directory is on the
lab host; a path that exists in one place does not exist in the other, and the failure is a
`FileNotFoundError` that looks like a missing run rather than a missing machine.

**The two best-track sources differ, and both are confirmed.** Training was centred on
**JTWC** — the intensities in the training filenames reach 170 kt in 2013, and JTWC's
1-minute sustained scale is the only one that reaches there (JMA's 10-minute scale put the
same storm at 125 kt). Inference uses **JMA**, verified by arithmetic: JMA has 202421W at
14.4 N on 2024-10-25 00Z, and `np.int_((90-14.4)/0.25)` snaps that to the 14.5 N the ERA5
file carries. The mismatch is real and bounded at ~30 km by the vortex-to-vortex agreement,
so it is a documentation item, not a research one.

`onnx/*.onnx` is gitignored (~106 MB) and must be copied separately.

**Copying to the lab host.** It does not listen on 22, so `scp` needs `-P 6606` — CAPITAL
`-P` for `scp`, while `ssh` takes lowercase `-p`, and lowercase `-p` on scp means "preserve
timestamps" and silently leaves you on port 22. `-O` forces the legacy SCP protocol that
OpenSSH 9 replaced with SFTP:

```
scp -O -P 6606 onnx/X.onnx onnx/X.yaml payne@140.112.67.82:~/LLAT_polar/onnx/
```

`tar cf - ... | ssh -p 6606 host 'cd ~/LLAT_polar && tar xf -'` avoids scp and sftp entirely.

### The Cartesian control, and what already exists

`/wk2/yungyun/FCNV2_TC/{TC_ID}/` holds finished coupled runs beside the IC, so **look before
running anything**: `one_way_couple_model`, `2_way_circle_couple_model`,
`2_way_circle_couple_model_LLATty_polar_v1`, several `*_v60_e3268` variants with DA and
nudging, `ERA5_bdy`, `DSAT_2D_obs`. A Cartesian season may already be there.

`{TC_ID}/ERA5/` holds `for_DLAMPty` (the `*_combined.nc` LLAT IC), `for_FCNV2` (the global
IC), and three track CSVs — `ERA5_TC_track.csv`, `ERA5_TC_track_new.csv`,
`ERA5_TC_track_radius5.csv`. Check which one a script wants; do not assume.

The Cartesian model runs from the **other repo**, `~/couple_FCNV2_LLAT`, with cwd there:

```
python inference_one_way_test.py --FCNV2_IC_path ... \
    --LLAT_IC_path .../{TC}_{stamp}_combined.nc --IC_time {stamp} \
    --save_folder ... --fore_hour 120 --LLAT_device cuda
```

`--LLAT_device` defaults to **cpu**, unusable for a season; pass `cuda`. One process peaks
near 9.2 GB, so run one at a time.

**Do not run the Cartesian model through `run_coupled_forecast.py`.**
`DLAMPty_inference.py:341` raises without a `polar:` block, and the boundary geometries are
irreducibly different (see Invariants). Merging the pipelines would hide the difference
rather than control for it.

## Throughput, measured 2026-08-21

Three guesses about why the GPUs sit at 41 % utilisation were made before
anything was measured, and all three were wrong. Recorded so they are not made
again.

| guess | how it died |
|---|---|
| the polar resampling is too slow on the CPU | `tools/dataloader_bench.py`: throughput flat from 4 workers up, 35.6 samples/s against 35.1 at twelve. A CPU-bound transform keeps scaling. |
| eighty dataloader workers oversubscribe twelve cores | `--cpus-per-task=12` is **per task**, so eight ranks have 96 cores, not 12. And sampling the running job gave idle-below-5 % for only **3 %** of samples — the GPUs are working 97 % of the time, not waiting. |
| the kernels are too small, so raise batch_size | measured, and it peaks at 16 then falls |

The sweep, 600 steps on 8 H200s at R=80/Theta=360, job IDs verified against the
`RunDir` line in each log rather than assumed from submission order:

```
batch    600 steps    per step    samples/s
    1        140 s      0.233 s          34
    2        142        0.237            68
    4        153        0.255           125
   16        410        0.683           187   <- throughput peak
   32        860        1.43            179
   64       1888        3.15            163
```

Below batch 4 the per-step time barely moves while the work quadruples, so a
fixed per-step cost — kernel launch and the DDP all-reduce — dominates and the
GPUs idle inside each step. Above 16 the step time grows superlinearly.

**There is no free speedup available from batch size, worker count or the
dataloader.** 41 % is what this model does on an H200: the window attention needs
many small reshapes and permutes, and launch overhead does not amortise. The only
untried lever is `torch.compile`, which fuses small ops and does not change the
optimisation.

What this does NOT settle is which batch size gives the best validation loss per
wall-clock hour, which is the question that actually matters. Steps and samples
pull opposite ways: batch 4 runs 12x more steps per hour than batch 64, batch 16
consumes 1.5x more samples per hour than batch 4. Every run so far used batch 4
and every gain came from more steps. Settle it with two jobs at equal `max_time`,
not equal `max_steps`.

Grid caching in `latlon_to_polar` is bitwise identical and not worth reporting as
a speedup: 31.3 against 29.1 samples/s, one sample each, inside a benchmark whose
own spread is +/- 15 %.

### Short experiments do not anneal

Exploration runs use `lr_scheduler_name: constant_warmup`. Only the final
production run uses `cosine`. `max_steps` defines the cosine curve, so with
annealing a job killed at the walltime leaves a rate that never finished decaying
and a model comparable to nothing, and two runs of different lengths cannot be
compared at the same step. Put it in the overlay, never in `config.yaml` — a
running job re-reads the config if it requeues.

The partition's limit is 48 h (`#SBATCH --time=2-00:00:00`, partition `8gpus`),
so a 30-hour run fits; `dev` is 4 h.

### Sixteen GPUs buy 4 %, and the batch sweep had already said so

2,000 steps, effective batch 32 both ways, `constant_warmup`:

```
 8 GPUs x batch 4   469 s   0.235 s/step   (25a-hgpn032)
16 GPUs x batch 2   452 s   0.226 s/step   (25a-hgpn016-017)   1.04x
```

The reason is in the sweep above: per-step time is 0.233 s at batch 1, 0.237 at
2 and 0.255 at 4 — nearly flat, because below batch 4 a step is dominated by
fixed cost, kernel launch and the gradient all-reduce, rather than by the work.
The 16-GPU per-step of 0.226 s is essentially the single-node batch-2 figure of
0.237. Crossing the interconnect cost almost nothing; there was simply nothing to
gain, because halving the per-device batch moves each GPU further into the regime
where the fixed cost dominates.

**More devices only help when each one has enough work to amortise the per-step
overhead.** Holding the effective batch fixed while doubling the devices
guarantees the opposite. Sixteen GPUs at batch 4 each would be effective batch
64, which is a different optimisation and not a speedup.

`torch.compile` remains the only untried lever. The first attempt died in an
Inductor kernel from symbolic-shape arithmetic; `dynamic=False` is the fix, since
every shape here is a constant.

### torch.compile: 2.3 %, inside the noise, left off

Separated from the per-step wall_time in the event files rather than from
sacct's total, which mixes in start-up. `train_loss_step` is logged every ~30
steps and each event carries a wall_time, so the steady rate and the one-off
compilation come apart:

```
          first interval   steady        logged span   sacct python
plain         0.211         0.217 s/step     430 s         469 s
compile       0.232         0.212 s/step     434 s         531 s
```

Compilation costs about 58 s - the difference between each run's sacct total and
its logged span, 97 s against 39 s - and it happens before the first logged
event. Steady state is 2.3 % faster, which breaks even at 11,600 steps and would
save roughly 35 minutes of a 25-hour run.

**Not worth using.** The two jobs ran on different nodes, and node-to-node spread
was measured at about 10 % elsewhere in this sweep, so 2.3 % is inside the noise
and may not be an effect at all. Confirming it would mean running both in one
allocation the way `job_scripts/sweep_batch.sh` does, and a 2.3 % ceiling does not
justify that. `COMPILE` stays unset; `dynamic=False` is required if it is ever
switched on, since the first attempt died in an Inductor kernel from
symbolic-shape arithmetic.

The method is worth keeping though: per-step timing lives in the event files
whether or not the progress bar is on, and `train_val_gap.read_events` already
returns it. Nothing needs rerunning to measure a step rate.

### The ceiling is the dataset, not the hardware

420,000 steps at 32 samples a step is 13.4 M sample presentations over 14,522
samples: **925 epochs**, and validation was still falling. On a dataset this small
against 24.6 M parameters, the gradient noise of a batch of 32 is plausibly doing
regularisation work, which is a reason to be careful about raising the batch: a
larger batch means a quieter gradient, less of that regularisation, and possibly
worse generalisation. That is a hypothesis, but it is consistent with the 24 % gap
and with the model being data-limited.

The directions that are not blocked by hardware:

- **more years** — training is 2007-2017 only
- **theta rotation as augmentation** — a roll of the array, free, and physically
  exact on a TC-centred polar grid. Cheaper than anything else on this list and
  unavailable to the Cartesian model.
- **rollout fine-tuning** — more supervision from the same samples

GraphCast and FourCastNet train on roughly 55,000 samples against this model's
14,522. No amount of batch tuning or extra devices closes that.
