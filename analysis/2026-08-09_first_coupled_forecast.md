# First coupled forecast — 202421W, init 2024-10-25 00Z, +24 h

Two questions answered, one hypothesis killed, one new defect located.

## Conclusions

1. **There is no outer-ring artefact in the coupled forecast.** Azimuthal spread
   rising toward the rim is what truth does too. The radial loss weight `w(r)=r^p`
   is not the priority it looked like.
2. **The frozen-boundary seam is real and coupling removes it.** Standalone puts a
   3.20 hPa spike in azimuthal MSLP spread at r = 8 where truth has 1.76;
   one-way has none.
3. **Track error is almost entirely along-track, and the storm moves at 62 % of
   observed speed.** Direction is right. This is a model deficiency, not a
   boundary one: coupling improves it by 7 %.
4. **New: the forecast inner core is more asymmetric than truth**, by 30 % at
   r = 1.9°, which is where P1 and B3 live.

## Evidence

### Radial profile, azimuthal std of MSLP (hPa)

`tools/radial_profile.py --lead 24`, spread taken about the axisymmetric part.

| | r=1.9 | r=5.1 | r=6.9 | r=8.0 | r=9.0 | r=9.4 |
|---|---|---|---|---|---|---|
| **ERA5** | **0.73** | **1.14** | **1.56** | **1.76** | **1.95** | **2.05** |
| one-way | 0.95 | 0.97 | 1.44 | 1.64 | 1.96 | 2.06 |
| standalone | 1.28 | 1.02 | 1.77 | **3.20** | 2.43 | 1.92 |

Beyond r = 5 one-way tracks truth to within 0.12 hPa and is very slightly
smoother, as an under-dispersive forecast should be. The rise toward the rim is
therefore real outer-domain weather — fronts, cloud bands — not an artefact.

**Without the ERA5 column this reads the opposite way.** A monotone rise toward
r_max is the published signature of an under-constrained outer domain, and
one-way shows exactly that shape. It is only wrong because truth shows the same
shape. Adding the reference curve took ten lines and changed the conclusion.

Standalone's spike at r = 8 is the boundary, frozen at t=0 and discontinuous with
the evolving interior. `--hold-radius` was 9°, so the peak sits just inside the
transition rather than on it.

The inner core goes the other way: at r = 1.9 truth is 0.73 and both forecasts
are higher. This is the first direct evidence that the coarse radial patch is
costing something. `patch_r = 8` at Δr 0.25° spans 2° while the radius of maximum
wind is 0.3–0.5°, so the whole core is tokenised as one patch (P1); the r = 0
singularity (B3) is in the same place.

### Track error (km), against the ERA5 domain centre

| lead | total | along | cross |
|---|---|---|---|
| 6 h | 104 | −101 | −25 |
| 12 h | 133 | −130 | −26 |
| 18 h | 164 | −164 | +2 |
| 24 h | 296 | −290 | −61 |

(one-way; standalone is 7 % worse at every lead.)

```
LLAT   144.50E, 14.50N -> 139.77E, 16.33N   =  516 km
ERA5   144.50E, 14.50N -> 137.00E, 16.50N   =  833 km
                                     ratio  =  62 %
```

Negative along-track throughout: the storm goes the right way, too slowly, by a
consistent factor. Cross-track stays under 61 km, so this is a speed bias rather
than a steering error — which matters, because the two need different fixes.

Error grows ~30 km per 6 h through 18 h, then nearly doubles between 18 and 24 h.
One point is not a trend; a longer forecast will say whether the storm accelerated
there or whether the growth is superlinear.

## Hypothesis for the speed bias — not yet tested

The Lagrangian frame encodes motion as a small perturbation on a large-mean
field. `lon` is normalised with std 12 (a hardcoded constant), so a 3 h
displacement of 0.94° is **0.078 σ**. Averaged over 20 surface channels, a model
that never moved the frame at all would pay under **0.4 %** of the surface L1
loss. The signal carrying storm motion sits near the noise floor of the
objective, and L1 rewards regression toward the mean — which is what a 62 % speed
looks like.

Two candidate fixes, if it holds: weight the coordinate channels in the loss, or
predict displacement rather than absolute position, which would put the signal at
O(1) σ but changes the data pipeline and the autoregressive closure.

**What this does not show.** The arithmetic bounds what the model *could* ignore;
it does not measure what it *does*. Testing it needs the per-step displacement
compared against truth, over more than one case.

## Limits of this note

One TC, one initial time, one lead range. The 62 % figure is a single case and
Kong-rey was fast-moving, which flatters a slow bias in absolute terms and
penalises it in relative ones. Nothing here separates the model from the ONNX
export or from the resampling, which the standalone/one-way agreement only
partially constrains.
