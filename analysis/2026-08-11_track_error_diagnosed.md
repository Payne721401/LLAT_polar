# The track error is the coordinate channels — 202421W, init 2024-10-25 00Z

Three independent measurements agree, and the roadmap changes as a result. Two
earlier hypotheses are dead, one inference-side repair is ruled out, and the
remaining lever is smaller than expected.

## The finding in one line

The model's own wind field is right; its declared frame moves at 55–72 % of what
that wind implies; and nothing in the objective asks it to do better, because the
two channels that carry the entire track are worth 0.43 % of the loss.

## Evidence

### 1. Single-step accuracy, from the training logs

Already recorded, never examined. `val_RMSE` is logged after de-standardisation
(`lightning_modules.py:173-176`), so it is in degrees.

| | final | in km at 20 °N |
|---|---|---|
| `val_RMSE/lon` | 0.774° | 81.0 |
| `val_RMSE/lat` | 0.554° | 61.6 |
| combined position error, one 3 h step | | **101.8** |
| distance Kong-rey covers in one 3 h step | | **104.0** |

Accumulated as a random walk this gives 288 km at 24 h. The observed one-way
error at 24 h was 296 km. **The coordinate channels account for the track error
on their own** — nothing else has to be wrong for the observed numbers to appear.

The share of the objective follows from the loss as assembled. `config.yaml` sets
no variable weights, so `weighted_loss` is False and the criterion is a plain
`nn.L1Loss`; `loss = upper + 0.25 × surface` over 78 and 20 channels
respectively. A surface channel is therefore worth 0.25/20 = 0.0125, against
1/78 = 0.0128 for an upper channel-level, and lon + lat together come to
**0.43 %** of the validation loss.

Two things fall out of the same curves. The coordinate error was **72.5 km at
step 2269** — better than the converged 102 km, so the model got worse at the one
thing that decides the track while getting better at everything else. And
checkpoint selection is not the culprit: over the converged tail the step chosen
by `val_loss` and the step best for lon differ by 0.0 km.

### 2. The steering flow — the model sees it and does not follow it

`tools/steering.py`, the paper's Fig. 15 applied to forecast output: areal mean
wind inside 500 km, 850–300 hPa deep-layer mean, against the storm's own motion.

| | median motion / steering |
|---|---|
| ERA5 | ≈ 1.05 |
| one-way | ≈ 0.72 |

Real storms move at roughly the speed of the flow they sit in, which is also a
check that the diagnostic works. The cleanest single point is hour 0, where the
initial condition is identical by construction:

```
steering   ERA5 8.8 m/s     one-way 8.9 m/s     same environment
motion     ERA5 10.3 m/s    one-way  5.7 m/s    55 %
```

Same input, same steering, half the response. The model's steering flow tracks
ERA5's to about +150 h, so **the environment representation is not the problem
for track** — this kills the hypothesis that the coarse outer-domain tokens (P1)
are what costs the track. P1 still matters for structure and intensity.

### 3. The vortex does not drift — no free repair

`tools/find_center.py`. If the storm slid away from the array centre, the drift
would be motion the model produced but failed to write into the coordinate
channels, and adding it back would correct the track without retraining.

| method | median offset | max |
|---|---|---|
| mslp | 30 km | 72 km |
| vort850 | 19 km | 197 km |
| wind_min | 28 km | 514 km |

The pressure centre stays within 1–1.5 grid cells of the array centre for the
whole 240 h, and vortex-against-vortex position error lies on top of
frame-against-frame at every lead. **The frame and the vortex lag together.**
There is nothing to recover.

The same table carries a warning about the definitions. `vort850` and `wind_min`
both fail at hour 0 — by 197 km and 459 km — on an ERA5 analysis, where the storm
position is not in doubt. The reason is in the best-track file: at initialisation
Kong-rey was **998 hPa and 35 kt**. A storm that weak has no eye for `wind_min` to
find. Use `mslp`.

## Best-track provenance, closed

The domain centre enters the system in exactly one place,
`download_ERA5_from_google_according_BT.py`, and nowhere else — neither
`DLAMPty_inference.py` nor either driver reads a position, only initial times.

That script snaps to the ERA5 grid with `np.int_`, which truncates the *index*.
Latitude is stored descending and longitude ascending, so a truncated index means
north on one axis and west on the other. Verified against the file:

```
JMA 2024-10-25 00Z      14.4 N, 144.5 E
(90 - 14.4)/0.25 = 302.4 -> 302 -> 14.5 N      +11 km north
     144.5 /0.25 = 578.0 -> 578 -> 144.5 E      exact
ERA5 file centre                14.50 N, 144.50 E   match
```

So the inference chain uses **JMA** (`TC_list_JMA_v2`), while the paper states the
training data was centred on **JTWC** positions and its reforecast verification
used IBTrACS. The mismatch is real but its size is now bounded: vortex-to-vortex
error, which does not depend on how the ERA5 files were centred, agrees with
frame-to-frame everywhere, so the ERA5 centres sit within about 30 km of their own
pressure minimum. Against errors of 300–2500 km this does not matter. It would
matter when comparing two models that differ by tens of km.

## What this case is not

The initialisation is a **35 kt, 998 hPa tropical storm**. The paper's Fig. 4b
case study is the same storm twelve hours later, when it was better organised, and
the paper reports track errors about 30 % larger for weak samples because the
vortex is poorly defined. Every number here is measured on the harder of the two
initialisations, and the comparison against the published figure is not
like-for-like.

## What follows

Ruled out as the leading cause of the track error: the lateral boundary (one-way
and standalone agree to +96 h), the environment representation (the steering flow
is right), vortex drift (there is none), the outer-ring artefact (truth shows the
same rise), and best-track provenance (bounded at 30 km).

What remains is the objective. Two changes, both needing a retrain:

- **`residual: true`** — already implemented and switched off
  (`pangu_polar.py:96,175-176,199-208`). Without it the network rebuilds the whole
  longitude ramp each step, and the signal that carries storm motion is 4.7 % of
  that ramp's own range. With it the network outputs only the change. It applies
  to all 98 channels, so a version masked to lon/lat alone is the more surgical
  form.
- **`surface_var_weights: {lon: ..., lat: ...}`** — raises their share of the
  objective from 0.43 %. Setting the key at all switches the criterion to
  `WeightedL1Loss`; unlisted variables take 1.0, so the other 18 are untouched.

And one inference-side test that needs no retraining: the direction is right —
cross-track error stayed under 250 km for 192 h — and only the speed is wrong, by
a factor near 1.45. Rescaling the per-step frame displacement measures how much of
the track error is pure speed calibration. It is a fitted constant on one case,
not a fix, but at a minute per forecast it is close to free.

## Limits

One storm, one initial time, and a weak one. The persistence baseline — the RMS
3-hourly displacement over the validation set — is still missing, and without it
"102 km per step" cannot be called worse than doing nothing, only compared against
this storm's own 104 km. The 1.45 factor is a median over one forecast.
