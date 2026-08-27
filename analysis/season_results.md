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

## 3. Intensity, four runs

Not yet run. Command:

```
python tools/season_intensity.py --era5-root /wk2/yungyun/FCNV2_TC \
  --runs "cart_1way=/wk2/yungyun/FCNV2_TC@one_way_couple_model" \
  --runs "cart_2way=/wk2/yungyun/FCNV2_TC@2_way_circle_couple_model" \
  --runs "polar_1way=/home/payne/LLAT_polar_runs_r80long_full@one-way" \
  --runs "polar_2way=/home/payne/LLAT_polar_runs_r80long_2way@two-way" \
  --max-lead 192 --every 24 \
  --out analysis/figures/season_intensity_4way.png
```

For reference, the one-way pair measured **-13.9 hPa** MSLP bias at +120 h
against the Cartesian run's -2.7, surviving the under-200 km track filter at
-10.8 hPa (n = 23). The paper's own bias is **+40 hPa** — too weak — for
best-track MSLP under 950 hPa, which is a conditional statistic and not
comparable to an all-cases mean. Stratifying is roadmap P3.

---

## 4. Provenance

- 2024 is complete in every root. The 2025 storms are not: `202504W`,
  `202506W` and `202518W` are present in some sweeps and absent from others,
  which is why the unpaired counts differ (one-way 338, two-way 339) while the
  intersection is 319.
- `2_way_circle_couple_model_v60_e3268` is a **different model** with DA and
  nudging, not the v57_5d control. Every Cartesian two-way number here comes
  from the unsuffixed `2_way_circle_couple_model`.
- Polar runs are `LLAT_polar_r80long_v1`, exported from `runs/t360_r80`
  (R = 80, Theta = 360, 420k steps, val 0.20042).
