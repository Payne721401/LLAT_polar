# LLAT polar — status and roadmap, 2026-08-23

`analysis/` is gitignored, so this file lives only on the machine that wrote it.
The durable summary is in `CLAUDE.md`; this is the working detail behind it.

---

## 1. Where the project is in one paragraph

Five models trained. The polar reformulation has closed most of the gap to the
Cartesian LLAT it replaces — track error went from 80 % worse to **6–8 % worse**
over a full 337-case 2024 season — and it solved the intensity failure the paper
identifies, but overshot: the paper reports strong TCs **+40 hPa too weak**, and
the polar model is now **too deep**. Per-variable RMSE is worse than Cartesian on
almost every field. The remaining levers are the loss function, data
augmentation, and the loss's spatial weighting — not hardware, and not more
steps.

---

## 2. Models trained

| run | grid | steps | val | gap | h | note |
|---|---|---|---|---|---|---|
| `prod_lr5e-5` | R41 patch(2,8,6) | 105k | 0.24997 | 15.2 % | 5.6 | baseline |
| `p1_wide` | R40 patch_r 4 | 105k | 0.24555 | 12.9 % | 5.6 | radial tokens 6→10 |
| `p1_theta360` | +Θ 360 | 105k | 0.24172 | 12.5 % | 6.7 | |
| `t360_long` | same | 210k | 0.22587 | 22.5 % | 13.4 | |
| `t360_pr2` | R40 patch_r 2 | 105k | 0.23354 | 10.6 % | 6.8 | |
| `t360_r80` | R80 patch_r 4 | 105k | 0.22766 | 11.9 % | 7.5 | |
| **`r80_420k`** | **R80 Θ360** | **420k** | **0.20042** | **33.3 %** | **30.2** | **best** |

Exported: `onnx/LLAT_polar_{vtvr,p1,t360,t360long,pr2,r80,r80long}_v1.{onnx,yaml}`.

### Diminishing returns, quantified

105k→210k bought 6.6 %; 184k→362k bought 3.0 %; 362k→420k bought **0.09 %**.
**About 300k steps is the knee** — the next run should stop there and save nine
hours.

### Patch geometry, measured

```
design              dr      patch_r    km    arc@1deg  arc@10deg   tokens
cartesian p(2,4,4) 0.25     1.00 deg  111    1.00 deg   1.00 deg     3528
t360  R40 pr=4     0.250    1.00      111    0.105      1.047        4800
r80   R80 pr=4     0.125    0.50       56    0.105      1.047        9600
R80 pr=2           0.125    0.25       28    0.105      1.047       19200
```

RMW is 0.3–0.5 deg. **Polar already beat Cartesian tenfold in azimuth near the
core; the radial patch was the binding constraint and r80 halved it.** patch_r
0.25 deg is one ERA5 cell — the floor for this data.

---

## 3. Season verification - SUPERSEDED, see analysis/season_results.md

Everything this section said was measured without clipping to the best-track
record and against ERA5 rather than best track, and both changed the answers.
The measured tables now live in `analysis/season_results.md`, which is tracked
in git. What changed, so an old number quoted elsewhere can be recognised:

| said here (2026-08-23) | actually |
|---|---|
| track: polar behind by **6-8 %** | **20-28 %** at 120-168 h once clipped |
| polar draws level at +168 h | it does not; that was both models scoring a dissipated system |
| intensity **-13.9 hPa**, "flipped the paper's sign" | withdrawn. TY is **+13.8**, still too weak, the same sign as the paper. Polar *reduces* the weak bias by 42 % |
| one all-cases intensity number | meaningless: TY +13.8 against TD -19.2 |

The cause of the first two: the ERA5 boxes were cut along a track that outlives
the agency record - 202408W's ran 3.5 days past its last entry, following a
1010.8 hPa remnant to 40N and the dateline - and 30-45 % of the scored leads
were in that gap.

New since, and not contradicted:

- **Two-way helps the Cartesian model and hurts the polar one**, over five
  consecutive leads: -14 % against +20 % at 120 h. Its harm is in the track and
  the domain-mean field, not the centre intensity.
- **A coherent hydrostatic drift** in the polar run: t850 -2.6 K, z500 -835,
  msl -465 Pa at +192 h, each roughly doubled by the feedback.
- **The polar model wins on q700 and t2m** at long lead.

## 11. Training probes and the radial picture (2026-09-02)

### The verification was wrong twice before any of this meant anything

Two corrections, in order, and every number predating them is superseded:

1. **Clip to the best-track record.** The ERA5 boxes were cut along a track
   that outlives the agency's - 202408W's ran 3.5 days past its last entry,
   following a 1010.8 hPa remnant to 40N and the dateline. 30-45 % of scored
   leads were in that gap and it hid a 20-28 % difference between the models.
2. **IBTrACS, not JMA, and not ERA5.** Truncation and best-track truth now come
   from IBTrACS, matched by position and time - the storm numbers are not
   interchangeable, 202405W is MARIA and WP052024 is GAEMI.

### The polar grid works. The polar training does not.

`tools/season_radial_rmse.py` splits every field by distance from the centre.
On an 81x81 box the ring inside 100 km is 45 cells of 6,561, so every domain
average in this project had been drowning the one statistic that tests the
formulation.

**At +24 h the inner core is ahead on six of ten variables** - u10 by 7.8 %,
t2m 6.3 %, q700 5.7 %, tp 5.1 %. **By +120 h it is behind on all ten**, msl by
54.8 % inside 100 km against 10.2 % beyond 600, and the inner-ring msl penalty
runs +0.9, +8.3, +20.3, +40.3, +54.7 % at 24 to 120 h. Monotonic from level,
which rules the grid out: a singularity at r = 0 would be there at +24 h and
constant. This accumulates.

The signed bias confirms it is core-centred rather than a whole-field offset -
the shapes are distinguishable and were checked on synthetic fields:

```
msl bias @120h [Pa]   0-100   100-300  300-600  600-1110
cart_1way             -230     -169     -106     -67       falls 3.5x
polar_1way           -1213     -768     -337     -107      falls 11x
z500  cart            -221     -190     -170     -197      FLAT: whole field
z500  polar           -859     -590     -345     -172      falls 5x
```

At the core the polar vortex is **deeper, warmer and moister** than truth
(msl -12 hPa, t850 +0.84 K, t2m +1.05 K, tcwv +4.7 kg/m2) where the Cartesian
is uniformly cold and dry. That is over-intensification, not an offset.

### Which truth: the two intensity results are both right

They disagreed because they use different truths, and the model sits between
them:

```
TY samples, core, +120 h      vs ERA5        vs IBTrACS
cart_1way                     -1.7 hPa        +23.9 hPa
polar_1way                    -9.3 hPa        +13.8 hPa
```

IBTrACS minus ERA5 is 23-25 hPa for typhoons, which is the paper's +40 hPa
cause - ERA5 cannot resolve an eyewall at 0.25 deg. **Both models are deeper
than ERA5 and shallower than reality, and the polar model's extra 7.6 hPa moves
it toward IBTrACS and away from its own training target.**

It helps typhoons and hurts everything weaker: against IBTrACS, polar is
**closer** for TY (+13.8 against +23.9) and **further** for TD (-19.2 against
-6.5) and TS (-13.5 against -3.9). One near-uniform 11 hPa shift, two opposite
consequences.

Error decomposition, TY core at +120 h: polar RMSE 1455 with bias -929, so
**41 % of its MSE is systematic**; Cartesian is 948 with bias -165, **3 %**.
The systematic part is the part a loss change can reach.

### Two-way: the mechanism is still unknown

The harm is in the **outer** rings, not the inner - msl +0.4 % inside 100 km
against +9.7 % beyond 600, rising to +41 % at +192 h. The earlier explanation
("it writes the bad core into FCNV2") is refuted by that. The feedback writes
outward inside 7.5 deg and FCNV2 writes back outside 8 deg, so the damage sits
where FCNV2 supplies LLAT. u10, tp and w500 are *better* with two-way.
**Untested**: dump the FCNV2 global field either side of the write-back.

Asymmetry is separately refuted (section 6).

### Training probes, 20k steps, constant LR, one variable each

First round ran at lr 4e-4 and all three diverged - config.yaml's value, which
no production run has ever used. 5e-5 everywhere; see section 10.

| probe | change | result |
|---|---|---|
| **`lossexcl`** | **8 prescribed channels out of the loss** | **won: vt10 -10.2 %, vr10 -8.5 %, msl -6.1 %, tp -6.0 %, tcwv -5.7 %, t2m -4.7 %; upper air level** |
| `residual_all` | residual, 10 channels excluded | lost on almost everything |
| `patch226` | patch_r 0.5 -> 0.25 deg | -0.84 % |
| `r160` | patch_r 0.126 deg | -1.66 %, 81 % slower |
| `lossexcl_patch` | lossexcl + patch + wide windows | lost to lossexcl by 1.6 %, 31 % slower |

**The geometry line is closed.** Three attempts, all negative, and the last one
had the receptive-field confound removed - windows widened 10 to 20 so a window
still spans 5.06 deg. No excuses left.

**loss_exclude is the first real win** and it costs nothing: the eight channels
are recomputed from the new centre and timestamp at every inference step, so
their predictions were never used. 44 % of the surface loss was being spent on
discarded outputs. Both the Cartesian and the polar model do this, so it is
inherited rather than a polar regression.

### Why the residual failed - not the reason expected

`tools/residual_scale.py`: sigma(3 h change) / sigma(field), which is the
standardised size of what a residual asks the network to emit. **Nothing is
below 0.1** - msl 0.51, u10 0.61, v10 0.63, t2m 0.39, tcwv 0.31, sst 0.13 at
the tightest (6-hourly boxes; the 3-hourly training data will read lower).
**"The residual target is too small to learn" is refuted.** The failure needs
another explanation, and `curve_table --ratio grad_2.0` against base is the
next place to look.

### Persistence

Model 101.8 km at one 3 h step against persistence's 68.4 km - the model
**loses** to "the storm does not move" at a single step. Read the tool's own
caveats: val_RMSE is a field RMSE while persistence's field and centre errors
are the same number, and one step is not a forecast. It is still a number worth
carrying.

---

## 12. The residual, and why the loss hypotheses are in trouble (2026-09-02)

### The residual connection: what was tried, what was ruled out

`residual_exclude` names the ten channels the residual skips - the eight
prescribed ones plus lon and lat. A residual is a prior on something the model
has to predict; the eight are recomputed from the new centre and timestamp at
every inference step, so a prior on them buys nothing however good identity
happens to be. lon and lat are the exception, genuinely used to place the next
domain, and excluded for the opposite reason: there identity means the storm
did not move, and this model already runs slow.

**Code checked, no defect.** The mask is applied once at Res Save so both
res_conn_after_smooth branches use it; the upper residual is deliberately
unmasked since no upper channel is prescribed; the mask is a non-persistent
buffer so it follows .to(device) and stays out of the state_dict; and input and
target share one normalisation, so target_std - input_std is the true change
divided by sigma and the residual is in the right units.

**It still lost**, on almost every surface variable, having started far ahead -
msl 171.89 against base's 480.08 at step 453 and level by 20,000.

Two explanations tested and both refuted:

- **"the residual target is too small to learn."** No.
  `tools/residual_scale.py` on the 3-hourly training data: msl 0.261, v10
  0.273, t2m 0.374, sp 0.351, mtnlwrf 0.453, tp 0.799, tightest is sst at
  0.167. **Nothing below 0.1.** The network is not being asked for a number
  beneath its own resolution.
- **"its gradients vanish."** Not straightforwardly. `curve_table --ratio
  grad_2.0` gives a median of 0.759 over 191 tags with 181 below one, so they
  are smaller - but structured rather than uniform: patch_recover.conv_upper
  0.429 and downsample.norm.weight 0.385 at the output and middle, while
  patch_embed.conv_upper.weight reads **1.800** and the layer1 blocks 1.05-1.53.
  The output head having less to do is expected; the input side getting *more*
  gradient is not explained, and the run's final error is larger than base's,
  which should have made every gradient bigger rather than smaller.

**So the residual's failure is unexplained.** It is cheap to leave alone and
that is the recommendation, but the two obvious causes are gone.

### The loss hypotheses do not survive the radial data

P1's stated rationale was that L1 tolerates a uniform offset. Both halves are
now contradicted:

1. **The offset is not uniform.** The msl bias falls 11x from core to rim
   (-1213 to -107 Pa). A uniform offset is flat, and the Cartesian z500 bias
   is what flat looks like: -221, -190, -170, -197.
2. **The Cartesian model uses the same loss** - unweighted L1, the same twenty
   channels, the same surface_weight - and carries 3 % bias against the polar
   model's 41 %. Identical loss, different outcome, so the loss alone is not
   the cause.

P5's area-weighting argument survived (2) - it is the one structural difference
between the grids, the core being over-weighted about 40x per unit area - but
runs into a harder problem:

**The core penalty is absent at +24 h.** msl inside 100 km: +0.9 % at 24 h,
then +8.3, +20.3, +40.3, +54.7 at 48 to 120. The loss is single-step. Eight
autoregressive steps in, the core is level; forty steps in it is 55 % worse.
**A single-step weighting - L1 against MSE, or w(r) - cannot produce an error
that only appears after tens of steps.**

What does act over tens of steps is **rollout drift**: the model is trained on
one step and asked for sixty-four, and a small systematic error compounds
fastest where the field's gradients are steepest, which is the core. That is
**P6, untouched**, and it is the only candidate whose mechanism matches both
the location and the timing.

**The discriminating measurement, one command:** the signed bias by ring at
+24, +48, +72, +96, +120 h (`season_radial_rmse --bias --lead L`). A core bias
already large at +24 h is a single-step problem and P1/P5 are the right
answers. A core bias near zero at +24 h that grows is rollout, and neither is.
The RMSE parity at +24 h says the second, but that is an inference from RMSE
and the bias has not been measured at short lead.

---

## 4. The paper, read (LLAT_paper.pdf, JAS-D-26-0056)

Facts that correct earlier assumptions in this project:

- **Training budget is "7 days on eight A100s"** (p15), not the 1,600,000 steps
  in `config.yaml` — that is a cap. On H200s that is roughly 2–3x r80_420k's 30
  hours, not the 3.8x claimed from the step count.
- **Loss is a weighted MSE** (p15), "following Bi et al. (2023)" — no independent
  justification. **The polar implementation uses L1**, in `lightning_modules.py`,
  with no comment anywhere explaining the change.
- **Data split is train 2007–2018, val 2019, test 2020**, centred on **JTWC**
  best track. The polar config uses train 2007–2017, val 2018–2019.
- **Evaluation (p18):** all 2024 WNP typhoons with Vmax > 65 kt — 26 storms, 00Z
  and 12Z, **319 initial times, 240 h**, truth is **IBTrACS position and MSLP**,
  baseline is **FourCastNetV2**, three boundary setups each.
- **Stated limitations (p46):** air–sea interaction and radiative feedback not
  explicitly represented; 0.25 deg limits eyewall dynamics and Vmax > 96 kt.
- **The key intensity result (p27):** for best-track MSLP below 950 hPa, LLAT.ty
  has a **positive bias of about 40 hPa** — too weak — attributed to ERA5's
  resolution. It still beats FCNV2 by 20–30 %.

**So the polar work fixed the paper's diagnosed cause (unresolved eyewall) and
flipped the sign of its main intensity error.** That is the cleanest causal chain
this project has, and it belongs in any report.

---

## 5. Ruled out — do not revisit without new evidence

Lateral boundary; domain size (best predictive steering radius is 4 deg, inside
the 10 deg disc); vortex drift (30 km median); outer-ring artefact as a coupling
product; best-track provenance (~30 km); vt/vr convention (0.0000 RMSE, three
years, both pairs); transform sign error; inverted meridional coupling; B3
cell-centred rings (7–15 % *worse*); `sst_filled` free-running (it tracks the
latitude cooling almost exactly, -14.5 against -15 expected).

**Hardware, all measured, all dead ends:** dataloader and `n_workers` (GPU idle
is 3 %); `batch_size` (throughput peaks at 12–16 then falls, and steps/hour drop
proportionally); 16 GPUs (**1.04x**); `torch.compile` (**2.3 %**, inside the
node-to-node noise, and it needs `dynamic=False` or it dies in an Inductor
kernel). 41 % GPU utilisation is what this model does on an H200.

---

## 6. Open findings, not yet explained

### Loss of asymmetric structure - REFUTED 2026-08-30

This was the leading hypothesis for both the track deficit and the two-way
harm, and it rested on one case at one lead: 202421W at +96 h, azimuthal
standard deviation of msl at r = 8 deg, 0.71 hPa against ERA5's 3.95. No tool
could reproduce it.

`tools/season_asymmetry.py` now measures it over the season, as the
wavenumber-1 amplitude on a ring divided by ERA5's at the same case and lead.
**The polar vortex is not too round.** In the core it is *more* asymmetric than
the truth (1.42 at +120 h, r = 2 deg), which fits a deeper core. At r = 8 deg it
sits 3-6 % below ERA5 between 48 and 120 h, where the Cartesian run sits 2-6 %
below - nothing like a factor of five, and far too small to explain a 20-28 %
track deficit. **And two-way matches one-way to within a few percent**, so the
feedback is not injecting an over-axisymmetric vortex into FCNV2.

**The two-way harm is unexplained.** It is real, large, consistent in sign over
five leads, and opposite between the two models, and no mechanism proposed so
far survives contact with a seasonal statistic.

### Outer-disc patch texture — largely gone in r80_420k

`polar_seam` on 202421W standalone, patch harmonic energy as a fraction of
azimuthal variance:

```
            inside 8 deg   outside
P1              0.19 %      1.23 %
r80_420k        0.01 %      0.08 %
```

**A fifteen-fold reduction.** The prediction recorded in `t360_long.yaml` — that
longer training would not remove it — was wrong twice over: it was
undertraining, and the finer radial grid helped too. Only +24 h has been checked;
longer leads may accumulate.

### Coordinate transform — priced, and not the culprit

Round trip on real ERA5, no model: msl and t2m 0.0 %, wind 0.7 %, **vorticity
4.2 %, precipitation 9.2 %, divergence 13.3 %**. Only 4 % of the error is a
function of radius alone, so it is interpolation on sharp gradients, not a ring
artefact. Θ 360 cuts it ~15 % versus Θ 180, all of it at the rim.

**Derivatives are amplified ~6x, so vorticity figures from a polar forecast show
a substantial amount of resampling. The fields that drive the track do not.**

---

## 7. Tools built this session

| tool | what it answers |
|---|---|
| `season_rmse.py` | per-variable, per-lead RMSE and bias vs ERA5, curves + a percentage bar panel |
| `season_intensity.py` | MSLP/wind error by lead, reported twice — all cases and track < 200 km |
| `landfall.py` | MSLP change in the N hours after the truth's landfall (**currently returns 0 cases — see §8**) |
| `train_val_gap.py` | train/val gap, best-checkpoint position, converged vs overfitting |
| `plot_curves.py` | val against steps and against compute hours, `--train` overlays training loss |
| `polar_seam.py` | patch-harmonic energy vs radius; refuses mixed standalone/one-way dumps |
| `transform_sensitivity.py` | what the round trip costs each field, derivatives computed after |
| `dataloader_bench.py` | dataloader throughput, no GPU needed |
| `terrain_check.py` | track over the model's own landmask, land and terrain per lead |
| `animate_forecast.py` | side-by-side GIF, frames keyed on content |
| `sweep_batch.sh` | all batch sizes in one allocation, no node-to-node noise |

Method worth keeping: **per-step timing is in the TFRecord event files whether or
not the progress bar is on**, and `train_val_gap.read_events` returns it. Nothing
needs rerunning to measure a step rate.

---

## 8. Known broken / immediate  (updated 2026-08-26)

All three items from 08-23 are now closed:

1. ~~`landfall.py` finds 0 landfalls~~ **Fixed.** `args.window` was a float, so
   `t0 + window` formatted as `126.0` in the filename and every one of the 260
   landfalls found was dropped at the load step, while a single counter reported
   it as "none found". Window is an int and the three drop reasons are counted
   separately. 231 cases now score. The boxplot's `labels=` keyword is handled
   with a try/except for the matplotlib rename to `tick_labels=`.
2. ~~neo82 driver mismatch~~ **Fixed.** Kernel and NVML are both 580.173.02,
   `torch.cuda.is_available()` is True on the RTX 3080, and `onnxruntime-gpu`
   1.28.0 replaced the CPU-only wheel, so `CUDAExecutionProvider` is available.
3. `t360_long`'s season still used `--max-starts 3` (81 cases). Unchanged.

### New, found 2026-08-26

- **`--llat-device` defaults to `cpu`** (`run_coupled_forecast.py`), so LLAT
  inference stays on the CPU no matter what onnxruntime can do. Pass
  `--llat-device cuda`. `--fcnv2-device` already defaults to `cuda` and raises
  when torch cannot see a GPU, so only the LLAT half was silent.
- **The provider message used to lie.** `DLAMPty_inference.load_model` printed
  the list it passed to `InferenceSession`, but ORT treats that as a preference
  and falls back to CPU without raising. It now prints `get_providers()` on the
  built session, so the message reports where inference actually runs.
- **The polar `config.yaml` has no `var_weights` at all** - not set, not even
  commented, unlike the Cartesian config which at least keeps them commented.
  With both weight dicts `None`, `lightning_modules.py` selects `nn.L1Loss`.
  **So the trained model is two steps from the paper, not one: unweighted, and
  L1 rather than MSE.** P1 below has to change both, and they should be changed
  one at a time.
- `season_stats.py --csv` and its head-to-head ranking now exist; the win rate
  beside the median is the number a median hides most.

## 9. Roadmap, in priority order  (revised 2026-08-30)

### P0 - extend probe_lossexcl to 105k (~8 h) - DO THIS FIRST

The only change that has won anything, and it is free. At 20k it beat base on
every surface variable while leaving the upper air alone. 20k is 4.8 % of a
production run, so the result needs a length that can be set beside
t360_r80's 0.22766 at 105k. Then rerun season_radial_rmse on it: the test is
whether the inner ring at +120 h moves back toward what it looks like at
+24 h.

Its val_loss is NOT comparable with base's - it averages twelve surface
channels against base's twenty, and the eight dropped are the easy ones. Use
val_RMSE per variable.

### P1 - L1 to weighted MSE, 105k steps (~7.5 h)

**Two changes, not one.** The polar `config.yaml` carries no `var_weights` at
all, not even commented, so with both weight dicts `None` `lightning_modules.py`
selects plain `nn.L1Loss`. The paper uses a weighted MSE. Change them one at a
time; MSE first, because the hypothesis is about the shape of the loss.

The evidence is a **near-uniform 11 hPa deepening** of the polar centre relative
to the Cartesian one, in every intensity group, alongside a -465 Pa domain-mean
msl bias. L1 is indifferent to a uniform offset of the whole field; MSE is not.

Judge it on the **stratified MSLP bias against best track**, not on validation
loss - the two losses are not comparable. The target is TY holding near +14
while TD and TS return towards zero.

**Open, and P1 does not answer it:** why only the polar model. The area-weighting
argument - every ring carries the same number of points while its area grows
with r, so an unweighted loss cares forty times more per unit area in the core
than at the rim - is polar-specific and would explain it, but P5 is not being
pursued (decision 2026-08-30). Expect P1 to reduce the offset without fully
accounting for its origin.

### P2 - theta rotation as augmentation

A roll along theta. Free, physically exact on a TC-centred polar grid, and
unavailable to the Cartesian model. It targets what actually limits this model:
14,522 samples and a 33 % train/val gap. Would be the polar formulation's first
structural advantage rather than a catch-up.

### P3 - DONE

`season_intensity --strat` splits on best-track Vmax into the paper's TD / TS /
TY groups, `--truth best` scores against IBTrACS, and `--strat-out` draws the
three-panel figure. The result is in `season_results.md` section 3 and it
changed the project's main intensity claim.

### P4 - match the paper's verification protocol - mostly done

Done: IBTrACS truth and truncation, the 26-storm 2024 sample (already
matching), 240 h on the Cartesian runs, the TD/TS/TY stratification.

**Left: FourCastNetV2 as the baseline.** The paper's headline is 25 % better at
120 h and 40 % at 144 h than FCNV2, and this project has no FCNV2 track to
compare against - the outputs were deleted to save disc. A rerun keeping at
least the FCNV2 centre track is what closes the gap.

### P5 - spatial loss weight w(r) = r^p - NOT BEING PURSUED

Decision 2026-08-30. Recorded because it is the only candidate that explains why
the offset appears in the polar model and not the Cartesian one; if P1 leaves a
residual offset, this is where it came from.

### P6 - rollout fine-tuning; P7 - seam check at 48/72/96 h

Unchanged, both still untouched.

### P8 - explain the two-way harm

New. Two-way costs the polar model 20 % at +120 h and buys the Cartesian model
14 %, over five consecutive leads. Asymmetry is refuted. Untested: the feedback
radius (7.5 deg in both implementations, the paper says 8), the sharpness of the
polar field where it is written back, and whether FCNV2's response to an 11 hPa
deeper vortex is simply outside its training distribution. Cheapest first test -
dump the FCNV2 global field before and after the write-back for one case and
look at what changed outside 8 degrees.

## 10. Standing decisions

- **Exploration runs use `constant_warmup`, not cosine.** `max_steps` defines the
  cosine, so a killed job is worthless and runs of different lengths cannot be
  compared at the same step. Only the final production run anneals.
- **Never edit `config.yaml` to test something** — a running job re-reads it on
  requeue. Use an overlay.
- **`--skip-done` makes a sweep resumable per case**, not per storm. Always pass
  it; `run_season.sh` does.
- **Judge a change by which direction of error it trades for which**, not by a
  median that averages two opposite errors. P1 traded false recurvatures for
  missed ones and the median did not move.
- **RMSE is a diagnostic, not a target.** It rewards blurring, and this model is
  already too smooth at long lead.
- **CRPS is not applicable** — it needs an ensemble, and the ten saved
  checkpoints are all within 390k–410k steps and far too correlated to serve as
  one.

- **IBTrACS is the best-track source**, for truncation and for truth. Not the
  JMA lists. The storm numbers are not interchangeable - JMA numbers named
  storms, JTWC numbers every depression - so 202405W is MARIA and WP052024 is
  GAEMI; match on time and position, never on the number.
- **Always pass `--clip-to-best-track`.** Without it 30-45 % of scored leads are
  against a system the agency stopped tracking, and that was large enough to
  hide a 20-28 % difference between the two models.
- **Never quote an all-cases intensity bias.** The TD and TY groups disagree in
  sign; their mean describes neither.
- **A number from one case is a hypothesis, not a result.** The asymmetry claim
  survived weeks on a single lead of a single storm and died in one seasonal
  pass. If a claim matters, it needs a tool.
