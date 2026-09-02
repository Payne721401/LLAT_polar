# Season verification — measured numbers

Tables only. Interpretation belongs in the status document; what is here is what
was measured, with the command that produced it, so a number can be re-derived
rather than trusted.

Tracked in git (`.gitignore` carries an exception for this one file) because
these are the project's results and they need to survive a machine change.

**Conventions.** Position error is the great-circle distance between the
forecast frame centre and the truth centre. Truth is ERA5 unless a table says
best track. Every table is scored on the initial times *all* of its runs have —
the count is in the heading. `n` falls with lead because storms end, and the
survivors are not a random subset, so a column at +192 h is a different, longer
lived population than one at +24 h and the two are not a skill trend.

---

## 1. Track, four runs, 319 paired cases

```
python tools/season_stats.py --era5-root /wk2/yungyun/FCNV2_TC \
  --runs "cart_1way=/wk2/yungyun/FCNV2_TC@one_way_couple_model" \
  --runs "cart_2way=/wk2/yungyun/FCNV2_TC@2_way_circle_couple_model" \
  --runs "polar_1way=/home/payne/LLAT_polar_runs_r80long_full@one-way" \
  --runs "polar_2way=/home/payne/LLAT_polar_runs_r80long_2way@two-way" \
  --worst 10 --worst-at 120 --out analysis/figures/season_4way.png \
  --csv analysis/season_4way.csv
```

Median position error, km:

| lead | n | cart_1way | cart_2way | polar_1way | polar_2way |
|-----:|--:|----------:|----------:|-----------:|-----------:|
|  24h | 296 |  81 |  81 |  85 |  84 |
|  48h | 263 | 144 | 151 | 182 | 189 |
|  72h | 222 | 247 | 237 | 263 | 282 |
|  96h | 177 | 358 | 324 | 371 | 394 |
| 120h | 131 | 452 | 389 | 477 | 570 |
| 144h |  92 | 623 | 548 | 633 | 718 |
| 168h |  61 | 935 | 862 | 923 | 979 |
| 192h |  35 | 1139 | 1336 | 1205 | 1409 |
| 216h |  14 | 1400 | 1462 | — | — |
| 240h |   6 | 1915 | 1893 | — | — |

Mean, km:

| lead | cart_1way | cart_2way | polar_1way | polar_2way |
|-----:|----------:|----------:|-----------:|-----------:|
|  24h | 102 | 102 | 107 | 108 |
|  48h | 208 | 211 | 227 | 235 |
|  72h | 336 | 341 | 347 | 365 |
|  96h | 500 | 500 | 496 | 542 |
| 120h | 665 | 617 | 730 | 796 |
| 144h | 903 | 776 | 954 | 1013 |
| 168h | 1308 | 1113 | 1298 | 1347 |
| 192h | 1602 | 1500 | 1664 | 1906 |

Worst single case, km:

| lead | cart_1way | cart_2way | polar_1way | polar_2way |
|-----:|----------:|----------:|-----------:|-----------:|
| 120h | 3043 | 3226 | 3457 | 4021 |
| 192h | 4587 | 4684 | 4814 | 6108 |

### Two-way against one-way, same runs

Change in median when the feedback is switched on. Negative is better.

| lead | Cartesian | polar |
|-----:|----------:|------:|
|  48h |  +5 % |  +4 % |
|  72h |  **-4 %** |  +7 % |
|  96h | **-10 %** |  +6 % |
| 120h | **-14 %** | **+20 %** |
| 144h | **-12 %** | **+13 %** |
| 168h |  **-8 %** |  +6 % |

Head to head at +120 h, 131 cases: `cart_2way` is closer in **76 of 131 (58 %)**.

The sign is opposite and holds over five consecutive leads. 48 h and earlier is
not informative — all four runs sit within 4 km of each other there. +192 h and
beyond has n ≤ 35 and should not be quoted.

---

## 1b. The same four runs, clipped to the best-track record

```
python tools/season_stats.py --era5-root /wk2/yungyun/FCNV2_TC   --track-csv /wk2/yungyun/ERA5_2024_for_TC/TC_list_JMA_v2 --clip-to-best-track   --runs ... (as above) --csv analysis/s_B.csv
```

Section 1 scores every lead that has an ERA5 box. Those boxes were cut along a
separate ERA5-derived track that outlives the agency record - 202408W's run 3.5
days past its last best-track entry, following a 1010.8 hPa remnant to 40N and
the dateline. This table scores only leads with a best-track record, which is
the rule the paper states.

Median position error, km. `n` is what survives the clip:

| lead | n | cart_1way | cart_2way | polar_1way | polar_2way |
|-----:|--:|----------:|----------:|-----------:|-----------:|
|  24h | 267 |  77 |  76 |  78 |  81 |
|  48h | 215 | 131 | 135 | 167 | 165 |
|  72h | 166 | 203 | 209 | 245 | 264 |
|  96h | 127 | 281 | 293 | 344 | 362 |
| 120h |  91 | 382 | 359 | 478 | 570 |
| 144h |  59 | 505 | 424 | 608 | 737 |
| 168h |  34 | 658 | 664 | 845 | 1052 |
| 192h |  20 | 1092 | 871 | 1708 | 2145 |

Between 30 and 45 % of the scored leads were beyond the best track — 131 cases
at +120 h become 91, 61 at +168 h become 34.

### This reverses the long-lead reading

polar_1way against cart_1way, median:

| lead | section 1 | clipped |
|-----:|----------:|--------:|
| 120h | +5.5 % | **+25.1 %** |
| 144h | +1.6 % | **+20.4 %** |
| 168h | **-1.3 %** (polar ahead) | **+28.4 %** |

**The polar model does not catch up at long lead.** Both models were being
scored against a dissipated system, and that noise is large enough to swamp the
difference between them. On the leads where a tropical cyclone still existed,
the polar model is 20-28 % behind throughout.

### Two-way, unchanged in sign and cleaner in size

Change in median when the feedback is switched on:

| lead | Cartesian | polar |
|-----:|----------:|------:|
| 120h |  **-6 %** | **+19 %** |
| 144h | **-16 %** | **+21 %** |
| 168h |  +1 % | **+24 %** |
| 192h | **-20 %** | **+26 %** |

Head to head at +120 h, 91 cases: `cart_2way` closer in **53 of 91 (58 %)** —
the same rate as the unclipped 76 of 131.

### The worst cases are different cases now

202408W is gone from every list: its peak was 35 kt and its whole contribution
came from leads after the agency stopped tracking it. What remains, in all four
runs, is **202410W** and **202405W** — genuinely hard cases rather than
verification artefacts.

+192 h still has n = 20 and should not be quoted.

---

## 2. Per-variable RMSE, one-way, 192 h

```
python tools/season_rmse.py --mode one-way --era5-root /wk2/yungyun/FCNV2_TC \
  --runs "cartesian=/wk2/yungyun/FCNV2_TC" \
  --runs "r80_420k=/home/payne/LLAT_polar_runs_r80long_full" \
  --max-lead 192 --every 24 --show-bias \
  --csv analysis/season_rmse_oneway_192.csv
```

RMSE against ERA5, storm-relative (both fields are on their own storm-following
grid, so the vortices align by construction and a track error shows up as a
different *environment*, not as a displaced vortex):

| variable | run | 24h | 48h | 72h | 96h | 120h | 144h | 168h | 192h |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| msl [Pa] | cartesian | 154 | 250 | 335 | 422 | 500 | 645 | 715 | 699 |
| msl [Pa] | r80_420k | 167 | 270 | 375 | 464 | 642 | 775 | 893 | **962** |
| z500 [m²/s²] | cartesian | 150 | 268 | 388 | 567 | 734 | 975 | 1219 | 1429 |
| z500 [m²/s²] | r80_420k | 179 | 303 | 440 | 523 | 791 | 1031 | 1356 | **1594** |
| t850 [K] | cartesian | 0.95 | 1.41 | 1.95 | 2.54 | 3.08 | 3.64 | 4.67 | 5.70 |
| t850 [K] | r80_420k | 1.08 | 1.58 | 2.14 | 2.56 | 3.21 | 3.84 | 5.11 | **6.44** |
| v850 [m/s] | cartesian | 2.92 | 4.00 | 4.86 | 5.54 | 6.14 | 6.58 | 6.99 | 7.38 |
| v850 [m/s] | r80_420k | 3.08 | 4.17 | 5.16 | 6.00 | 7.26 | 7.84 | 8.59 | **8.99** |
| u850 [m/s] | cartesian | 2.91 | 4.12 | 5.17 | 6.19 | 6.87 | 8.19 | 9.67 | 10.67 |
| u850 [m/s] | r80_420k | 3.07 | 4.30 | 5.41 | 6.30 | 7.75 | 8.99 | 9.87 | 10.88 |
| u10 [m/s] | cartesian | 2.07 | 2.91 | 3.63 | 4.29 | 4.79 | 5.34 | 5.92 | 6.14 |
| u10 [m/s] | r80_420k | 2.12 | 2.90 | 3.56 | 4.14 | 4.89 | 5.42 | 5.88 | 6.25 |
| v10 [m/s] | cartesian | 2.14 | 2.99 | 3.70 | 4.14 | 4.44 | 4.69 | 5.06 | 5.19 |
| v10 [m/s] | r80_420k | 2.26 | 3.05 | 3.76 | 4.25 | 4.94 | 5.17 | 5.56 | 5.92 |
| t2m [K] | cartesian | 1.30 | 1.88 | 2.38 | 3.09 | 3.52 | 4.30 | 5.76 | 6.76 |
| t2m [K] | r80_420k | 1.44 | 2.20 | 2.88 | 3.36 | 3.92 | 4.49 | **5.29** | **6.03** |
| q700 [kg/kg] | cartesian | 1.365e-3 | 1.763e-3 | 2.090e-3 | 2.311e-3 | 2.408e-3 | 2.541e-3 | 2.907e-3 | 3.216e-3 |
| q700 [kg/kg] | r80_420k | 1.400e-3 | 1.794e-3 | 2.025e-3 | **2.164e-3** | **2.388e-3** | 2.576e-3 | **2.745e-3** | **3.091e-3** |
| tcwv [kg/m²] | cartesian | 4.68 | 6.39 | 7.89 | 9.10 | 9.81 | 10.43 | 12.15 | 13.21 |
| tcwv [kg/m²] | r80_420k | 5.05 | 6.79 | 7.94 | 8.97 | 10.26 | 11.41 | 12.44 | 13.61 |
| tp [m] | cartesian | 9.90e-4 | 1.100e-3 | 1.189e-3 | 1.234e-3 | 1.276e-3 | 1.298e-3 | 1.270e-3 | 1.194e-3 |
| tp [m] | r80_420k | 1.005e-3 | 1.117e-3 | 1.247e-3 | 1.382e-3 | 1.566e-3 | 1.507e-3 | 1.492e-3 | 1.488e-3 |

Bold marks where the polar model is better.

### Bias, and how much of the error is systematic

Bias against ERA5. The polar run's msl, z500 and t850 all start slightly
positive, cross zero between 48 and 72 h, and then fall together — one coherent
hydrostatic drift, not three independent errors: the column cools, the thickness
drops, the height falls, the surface pressure falls.

| variable | run | 24h | 96h | 120h | 192h |
|---|---|--:|--:|--:|--:|
| msl [Pa] | cartesian | -9.8 | -55.9 | -86.3 | -115.3 |
| msl [Pa] | r80_420k | +16.0 | -121.5 | -228.6 | **-443.3** |
| z500 [m²/s²] | cartesian | -24.7 | -109.0 | -189.6 | -330.6 |
| z500 [m²/s²] | r80_420k | +11.9 | -109.8 | -255.8 | **-805.6** |
| t850 [K] | cartesian | -0.069 | -0.193 | -0.496 | -1.560 |
| t850 [K] | r80_420k | +0.051 | +0.102 | -0.160 | **-2.523** |
| v10 [m/s] | cartesian | +0.057 | +0.367 | +0.395 | +0.323 |
| v10 [m/s] | r80_420k | +0.134 | +0.705 | +0.801 | **+0.867** |
| t2m [K] | cartesian | -0.058 | -0.428 | -0.808 | -2.298 |
| t2m [K] | r80_420k | +0.033 | +0.669 | +0.616 | **-1.086** |

Systematic share of the msl error, |bias| / RMSE at +192 h:

| run | RMSE | \|bias\| | share |
|---|--:|--:|--:|
| cartesian | 699 | 115 | 16 % |
| r80_420k | 962 | 443 | **46 %** |

The meridional wind is consistently worse than the zonal — u10 is level with the
Cartesian run while v10 is 6–14 % worse, and v850 is 5–22 % worse against u850's
5–12 %. vt/vr to u/v is a pure rotation and cannot produce that asymmetry, so it
is not the coordinate transform.

---

## 3. Intensity, against best track, split by intensity group

```
python tools/season_intensity.py --era5-root /wk2/yungyun/FCNV2_TC   --ibtracs /home/payne/ibtracs/ibtracs.WP.list.v04r01.csv   --clip-to-best-track --truth best --strat --max-lead 192 --every 24
```

**Truth is the IBTrACS central pressure**, not ERA5's own minimum. The two are
not interchangeable: ERA5 cannot resolve an eyewall at 0.25 degrees and is
systematically too weak for intense storms, which is the source of the paper's
+40 hPa. Positive bias means the forecast is **too weak**.

MSLP bias [hPa], by best-track Vmax **at the forecast time**:

| lead | | TD <35 | | TS 35-65 | | TY >=65 | |
|-----:|---|---:|---:|---:|---:|---:|---:|
| | | cart_1way | polar_1way | cart_1way | polar_1way | cart_1way | polar_1way |
|   0h | n 82/130/90 | +2.3 | +2.3 | +5.1 | +5.1 | +24.4 | +24.4 |
|  24h | | +1.8 | +1.5 | +4.7 | +3.4 | +24.5 | **+20.8** |
|  48h | | 0.0 | -1.3 | +1.6 | -0.6 | +25.0 | **+20.6** |
|  72h | | -1.9 | -7.0 | -1.3 | -6.1 | +25.8 | **+19.3** |
|  96h | | -4.4 | -12.9 | -2.0 | -9.5 | +24.1 | **+16.0** |
| 120h | n 22/28/34 | -6.5 | -19.2 | -3.9 | -13.5 | +23.9 | **+13.8** |

At +0 h all four runs and all three groups agree — that is the initial
condition, and it is already 24 hPa too weak for typhoons. Everything below it
is the model's doing.

### It is one near-uniform offset, not an intensity-dependent error

polar_1way minus cart_1way at +120 h:

| group | difference |
|---|---:|
| TD | -12.7 hPa |
| TS | -9.6 hPa |
| TY | -10.1 hPa |

**The polar centre is about 11 hPa deeper than the Cartesian one at the same
lead, whatever the storm's strength.** For typhoons that closes 42 % of a weak
bias the paper attributes to unresolved eyewalls; for weak systems it pushes a
near-zero bias to -19. One offset, two opposite consequences.

**The earlier claim that this project reversed the paper's intensity bias is
withdrawn.** The TY column is still positive — still too weak, the same sign the
paper reports. What the polar formulation does is *reduce* that bias, which is
a smaller claim and a defensible one, and it connects directly to the cause the
paper names.

**An all-cases mean is not usable here.** Averaging a +13.8 against a -19.2
gives -14.6, which describes neither group; every earlier figure quoted in that
form should be read as this table instead.

### Two-way barely touches the centre

polar_2way against polar_1way at +120 h: TD -18.7 against -19.2, TS -14.5
against -13.5, TY +17.3 against +13.8. The feedback's harm, established in
section 1b, is in the **track and the domain-mean field**, not in the eye.

---

## 3b. Vortex asymmetry — the proposed mechanism, refuted

```
python tools/season_asymmetry.py --era5-root /wk2/yungyun/FCNV2_TC   --ibtracs ... --clip-to-best-track --max-lead 192 --every 24
```

Azimuthal wavenumber-1 amplitude of msl on a ring, as a fraction of ERA5's at
the same case and lead. Below 1 is too axisymmetric.

| r | run | 48h | 96h | 120h | 192h |
|--:|---|---:|---:|---:|---:|
| 2 deg | cart_1way | 1.03 | 1.22 | 1.11 | 1.34 |
| 2 deg | polar_1way | 1.05 | 1.16 | **1.42** | **1.57** |
| 8 deg | cart_1way | 0.98 | 0.94 | 1.09 | 1.20 |
| 8 deg | polar_1way | 0.95 | 0.91 | 0.97 | 1.20 |
| 8 deg | polar_2way | 0.93 | 0.94 | 0.96 | 1.15 |

**The polar vortex is not too round.** In the core it is *more* asymmetric than
the truth, increasingly so with lead — consistent with a deeper core, which has
larger gradients for any distortion to act on. At 8 degrees it is 3-6 % below
ERA5 between 48 and 120 h, and the Cartesian run is 2-6 % below over the same
span: a difference far too small to produce a 20-28 % track deficit.

**And two-way does not reduce it.** polar_2way tracks polar_1way to within a few
percent at every radius and lead. The hypothesis that the feedback injects an
over-axisymmetric vortex into FCNV2 and corrupts the steering **does not
survive**; the two-way harm is still unexplained.

The single-case figure this project carried — an azimuthal standard deviation
one fifth of ERA5's, from 202421W at +96 h — **does not generalise** and should
not be quoted.

---

## 3c. RMSE by distance from the centre - the grid works, the training does not

```
python tools/season_radial_rmse.py --era5-root /wk2/yungyun/FCNV2_TC   --ibtracs ... --clip-to-best-track --max-lead 192 --every 24 --lead 120
```

Every RMSE above is a domain average. On an 81x81 box the ring inside 100 km is
45 cells of 6,561 - seven tenths of one percent - so the statistic that decides
whether the polar formulation works at all was being drowned by the region
where it makes no claim.

### At +24 h the inner core is ahead on six of ten variables

polar_1way against cart_1way, 0-100 km, +24 h:

| better | | worse | |
|---|--:|---|--:|
| u10 | **-7.8 %** | z500 | +12.6 % |
| t2m | **-6.3 %** | v10 | +1.3 % |
| q700 | **-5.7 %** | w500 | +1.1 % |
| tp | **-5.1 %** | msl | +0.9 % |
| t850 | -1.2 % | | |
| tcwv | -0.8 % | | |

**The resolution advantage is real and it appears exactly where the grid was
designed to buy it.**

### By +120 h it is gone, and the loss grows from the centre outward

| ring | msl | z500 | t850 | tcwv | q700 |
|---|--:|--:|--:|--:|--:|
| 0-100 km | **+54.8 %** | **+35.0 %** | **+21.6 %** | **+21.5 %** | **+18.2 %** |
| 100-300 | +59.9 % | +23.9 % | +8.9 % | +15.7 % | **-6.4 %** |
| 300-600 | +34.1 % | +14.4 % | +1.2 % | +7.1 % | **-5.3 %** |
| 600-1110 | +10.2 % | +0.1 % | +5.1 % | +2.1 % | +1.2 % |

msl in the inner ring, by lead: **+0.9 % at 24 h, +8.3 at 48, +20.3 at 72,
+40.3 at 96, +54.7 at 120.** Monotonic from level.

**That rules out the grid as the cause.** A singularity at r = 0, or the
interpolation onto the polar mesh, would be there at +24 h and roughly constant
after. This accumulates, which is what an error in the model's own dynamics
does.

### It unifies everything else in this document

- the **11 hPa centre offset** (section 3) is what an inner-ring msl penalty
  looks like measured at a point;
- the **hydrostatic drift** (z500 +35 % inner against +0.1 % outer) is the same
  error at a different level;
- **two-way harm** (section 1b): the feedback writes back inside 7.5 deg =
  833 km, which covers the three rings where the polar field is 34-60 % worse.
  It is putting the worst part of the polar field into FCNV2. That also predicts
  the radius cannot be tuned out of trouble - polar is worse in every ring, only
  less so at the rim;
- **q700** is the one variable that wins in the middle rings, and the only one
  not dominated by the centre error.

**So the polar grid is not the problem and the polar training is.** Those are
separable, and the second is P1.

**P1's acceptance test is now concrete**: after the loss change, the inner ring
at +120 h should look like the inner ring at +24 h does today - six of ten
variables ahead - rather than ten of ten behind. That is a far sharper target
than a validation loss.

---

## 4. Provenance

- 2024 is complete in every root. The 2025 storms are not: `202504W`,
  `202506W` and `202518W` are present in some sweeps and absent from others,
  which is why the unpaired counts differ (one-way 338, two-way 339) while the
  intersection is 319.
- **The 2024 sample is 26 storms in both the one-way and the two-way sweep,
  which is exactly the paper's 26.** The paper describes its selection as "all
  2024 typhoons, i.e. with maximum intensity (Vmax) greater than 65 kt", but
  that criterion does not reproduce its own count: IBTrACS v04r01 has 15 WP
  storms reaching 65 kt on `USA_WIND` in 2024 and 13 on `WMO_WIND`. 26 is the
  number of named 2024 WNP storms. So the samples already match and
  `--min-lifetime-vmax 65` would cut this set to 15 and move it *away* from the
  paper's. Do not apply it.
- **JMA and IBTrACS end their records within 0-18 h of each other**, so
  clipping on JMA is equivalent to clipping on IBTrACS, which is what the paper
  does. Checked by matching each JMA storm to the IBTrACS storm it overlaps
  most; for the unambiguous matches - SHANSHAN, YAGI, JEBI, KRATHON, TRAMI,
  KONG-REY, MAN-YI - the end differs by 0, +6 or +12 h, one or two 6-hourly
  records out of about thirty. IBTrACS *starts* 12-48 h earlier because JTWC
  numbers a depression before JMA names it, but that only changes which initial
  times exist, not where a forecast's leads are cut.
  Note the storm numbers are not interchangeable: JMA numbers named storms and
  JTWC numbers every depression it tracks, so they diverge during the season
  and `202405W` is not `WP052024`. Match on time overlap, never on the number.
- Verification truth is JMA (`TC_list_JMA_v2`); the paper's is IBTrACS. For
  position this is not worth changing: the WMO position for the WNP comes from
  RSMC Tokyo, i.e. JMA, and the ERA5 box centres were measured at about 30 km
  from their own pressure minimum against errors of 300-2500 km. It would
  matter for intensity, where the agency and the averaging period differ
  (JMA 10-minute against JTWC 1-minute, roughly 12 % apart).
- **ERA5 boxes outlive the best track.** 202408W's JMA record ends 2024-08-15
  18Z; its `for_DLAMPty` boxes run to 08-19 06Z, cut along a separate
  ERA5-derived track that follows a 1010.8 hPa remnant to 40N and the dateline.
  Every table above without `--clip-to-best-track` scores those leads.
- `2_way_circle_couple_model_v60_e3268` is a **different model** with DA and
  nudging, not the v57_5d control. Every Cartesian two-way number here comes
  from the unsuffixed `2_way_circle_couple_model`.
- Polar runs are `LLAT_polar_r80long_v1`, exported from `runs/t360_r80`
  (R = 80, Theta = 360, 420k steps, val 0.20042).
