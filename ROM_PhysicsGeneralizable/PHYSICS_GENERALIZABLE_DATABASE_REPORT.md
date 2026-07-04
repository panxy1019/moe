# Physics-Generalizable 2D Cylinder ROM Database

Generated/updated: 2026-07-04 21:32:54

## Goal

This independent data-generation branch targets physics-generalizable parametric ROM training, not only interpolation accuracy in one Reynolds-number interval.  The database covers `Re=20-200`, spanning stable steady wake, the Hopf onset near `Re≈46-47`, developing periodic vortex shedding, mature two-dimensional periodic shedding, and the upper two-dimensional range before the real cylinder wake develops Mode-A three-dimensional instability.

Existing data and scripts are not modified.  Branch path:

`/home/ray/physics_generalizable_rom_re20_200_branch`

Dataset path:

`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re`

## Physical Basis

- The circular-cylinder wake onset is treated as a supercritical Hopf bifurcation near `Re≈46-47`.
- Three-dimensional secondary instability appears near `Re≈188.5` with Mode A, and another branch appears near `Re≈259`; therefore this physics-generalizable 2D database stops at `Re=200`.
- References used for this design:
  - Williamson, C. H. K. (1996), "Vortex Dynamics in the Cylinder Wake", Annual Review of Fluid Mechanics, DOI: https://doi.org/10.1146/annurev.fl.28.010196.002401
  - Barkley & Henderson (1996), "Three-dimensional Floquet stability analysis of the wake of a circular cylinder", Journal of Fluid Mechanics, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/threedimensional-floquet-stability-analysis-of-the-wake-of-a-circular-cylinder/61575FBF0BC45054592D46382DEF30BB
  - Noack & Eckelmann (1994), "A global stability analysis of the steady and periodic cylinder wake", Journal of Fluid Mechanics.

## Reynolds Sampling

The 100 Reynolds numbers are generated automatically from a smooth density function rather than a fixed hand-written list.  The density is highest in `40-60`, high in `20-40` and `60-100`, moderate-high in `100-150`, and lighter but still covering `150-200`.

Regime counts:

{
  "steady_wake": 11,
  "pre_hopf_steady": 9,
  "hopf_transition": 17,
  "developing_periodic_shedding": 24,
  "mature_periodic_shedding": 22,
  "high_re_2d_periodic_near_modeA": 17
}

Actual Re distribution:

```text
000: 20.000000  steady_wake
001: 22.535676  steady_wake
002: 24.630436  steady_wake
003: 26.667332  steady_wake
004: 28.695138  steady_wake
005: 30.720428  steady_wake
006: 32.740068  steady_wake
007: 34.737570  steady_wake
008: 36.657767  steady_wake
009: 38.357249  steady_wake
010: 39.685479  steady_wake
011: 40.711525  pre_hopf_steady
012: 41.576575  pre_hopf_steady
013: 42.359071  pre_hopf_steady
014: 43.093925  pre_hopf_steady
015: 43.797402  pre_hopf_steady
016: 44.478353  pre_hopf_steady
017: 45.142703  pre_hopf_steady
018: 45.795194  pre_hopf_steady
019: 46.440072  pre_hopf_steady
020: 47.081355  hopf_transition
021: 47.722947  hopf_transition
022: 48.368688  hopf_transition
023: 49.022357  hopf_transition
024: 49.687640  hopf_transition
025: 50.368054  hopf_transition
026: 51.066785  hopf_transition
027: 51.786450  hopf_transition
028: 52.528767  hopf_transition
029: 53.294175  hopf_transition
030: 54.081508  hopf_transition
031: 54.887950  hopf_transition
032: 55.709610  hopf_transition
033: 56.543246  hopf_transition
034: 57.389970  hopf_transition
035: 58.262636  hopf_transition
036: 59.201432  hopf_transition
037: 60.307745  developing_periodic_shedding
038: 61.755954  developing_periodic_shedding
039: 63.499817  developing_periodic_shedding
040: 65.259829  developing_periodic_shedding
041: 66.970112  developing_periodic_shedding
042: 68.649714  developing_periodic_shedding
043: 70.314635  developing_periodic_shedding
044: 71.972931  developing_periodic_shedding
045: 73.628287  developing_periodic_shedding
046: 75.282340  developing_periodic_shedding
047: 76.935803  developing_periodic_shedding
048: 78.588971  developing_periodic_shedding
049: 80.241943  developing_periodic_shedding
050: 81.894708  developing_periodic_shedding
051: 83.547164  developing_periodic_shedding
052: 85.199103  developing_periodic_shedding
053: 86.850168  developing_periodic_shedding
054: 88.499815  developing_periodic_shedding
055: 90.147341  developing_periodic_shedding
056: 91.792204  developing_periodic_shedding
057: 93.435204  developing_periodic_shedding
058: 95.081752  developing_periodic_shedding
059: 96.749308  developing_periodic_shedding
060: 98.480345  developing_periodic_shedding
061: 100.352251  mature_periodic_shedding
062: 102.440042  mature_periodic_shedding
063: 104.710911  mature_periodic_shedding
064: 107.050737  mature_periodic_shedding
065: 109.395985  mature_periodic_shedding
066: 111.734011  mature_periodic_shedding
067: 114.066308  mature_periodic_shedding
068: 116.395488  mature_periodic_shedding
069: 118.723173  mature_periodic_shedding
070: 121.050171  mature_periodic_shedding
071: 123.376833  mature_periodic_shedding
072: 125.703274  mature_periodic_shedding
073: 128.029461  mature_periodic_shedding
074: 130.355225  mature_periodic_shedding
075: 132.680203  mature_periodic_shedding
076: 135.003744  mature_periodic_shedding
077: 137.324830  mature_periodic_shedding
078: 139.642302  mature_periodic_shedding
079: 141.956319  mature_periodic_shedding
080: 144.273459  mature_periodic_shedding
081: 146.619578  mature_periodic_shedding
082: 149.059229  mature_periodic_shedding
083: 151.686208  high_re_2d_periodic_near_modeA
084: 154.520852  high_re_2d_periodic_near_modeA
085: 157.459588  high_re_2d_periodic_near_modeA
086: 160.415176  high_re_2d_periodic_near_modeA
087: 163.364123  high_re_2d_periodic_near_modeA
088: 166.306373  high_re_2d_periodic_near_modeA
089: 169.244893  high_re_2d_periodic_near_modeA
090: 172.181708  high_re_2d_periodic_near_modeA
091: 175.117940  high_re_2d_periodic_near_modeA
092: 178.054368  high_re_2d_periodic_near_modeA
093: 180.992055  high_re_2d_periodic_near_modeA
094: 183.933395  high_re_2d_periodic_near_modeA
095: 186.884600  high_re_2d_periodic_near_modeA
096: 189.862278  high_re_2d_periodic_near_modeA
097: 192.911664  high_re_2d_periodic_near_modeA
098: 196.160723  high_re_2d_periodic_near_modeA
099: 200.000000  high_re_2d_periodic_near_modeA
```

The same values are stored in `Re_sampling_strategy.csv`.

## Steady And Periodic Criteria

For `Re < 47`, the workflow first solves the stable steady wake with `simpleFoam` using residual, force, and volume-statistic monitoring.  This avoids contaminating the steady-wake regime with non-physical startup oscillations from an impulsive transient run.  The retained data are still written by restarting `pimpleFoam` from the steady field, so the saved files remain consistent with the transient OpenFOAM data path.  A case enters its retained-save window after:

- lift coefficient standard deviation is small,
- drag coefficient relative variation and trend are small,
- recent solver final residuals are below tolerance,
- volume-averaged field statistics are stable when available, or the steady solver has completed its conservative iteration cap and the exact diagnostics are recorded.

For `Re >= 47`, the workflow monitors lift coefficient peaks.  A stable limit cycle is accepted when recent peak amplitudes and peak-to-peak periods have small coefficient of variation after at least five estimated shedding periods.  At least 20 detected shedding periods are retained.

## Storage Flow

Each case runs in a temporary branch-local work directory.  After `pimpleFoam`:

1. `reconstructPar`
2. `foamToVTK -useTimeName -noFaceZones`
3. convert retained VTK point fields to compressed `.npz`
4. verify the `.npz`
5. delete VTK and temporary OpenFOAM work directory

## Area-Weighted POD

The POD uses lumped nodal 2D control areas.  OpenFOAM mesh cell volumes are exported through VTK and divided by the mesh thickness to form cell areas.  Point areas are assembled by distributing each adjacent cell area equally to that cell's vertices:

```text
A_i = sum_{cell c containing point i} Area(c) / N_vertices(c)
```

Velocity and pressure snapshots are weighted by `sqrt(A_i)` before SVD:

```text
X_w = sqrt(A) * X
Y_w = sqrt(A) * Y
```

Thus Euclidean snapshot products correspond to the finite-volume discrete L2 inner products:

```text
<q_a, q_b> = sum_i A_i (u_a,i u_b,i + v_a,i v_b,i)
<p_a, p_b> = sum_i A_i p_a,i p_b,i
```

The POD implementation is streaming/blockwise: it does not write a full weighted snapshot matrix to disk, which prevents disk exhaustion and avoids the large dense temporary arrays that caused problems in older runs.

## Current Status

- Final dataset completed: `True`
- Completed/skipped case count recorded in this report run: `100`
- Current free disk: `8.35 GiB`
