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
