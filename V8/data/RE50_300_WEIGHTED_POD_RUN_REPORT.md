# Re=50-300 Cylinder Dataset And Weighted POD Run Report

Date: 2026-06-16

## Physical Regime Check

Classical circular-cylinder wake literature places the steady-to-periodic Hopf bifurcation near `Re_c ≈ 46-47`; therefore the requested lower bound `Re=50` is just above the vortex-shedding onset.

The real cylinder wake becomes three-dimensional around `Re≈180-200`, with further mode changes in the `Re≈230-260` range. Therefore two-dimensional simulations above about `Re≈200`, and especially near `Re=300`, are best interpreted as two-dimensional Navier-Stokes benchmark data or ROM training data, not quantitatively faithful 3D laboratory cylinder-wake data.

## Sampling

- Re range: `50-300`
- Number of Re points: `100`
- Dense segment: 60 nonuniform points from 50 to 120
- Sparse segment: 40 nonuniform points from 120 to 300
- First Re: `50.000000`
- Last Re: `300.000000`

## Time Window

- Estimated Strouhal formula: `St = max(0.05, 0.198 * (1 - 19.7/Re))`
- Total simulated cycles: `20`
- Dropped transient cycles: `4`
- Retained cycles in NPZ/POD: `16`
- Frames per cycle target: `8`
- Retained frames per Re target: `129`

## Solver

- OpenFOAM solver: `pimpleFoam`
- Momentum model: laminar
- Kinematic viscosity: `nu = 0.001 m^2/s`
- Cylinder diameter: `D = 1 m`
- Inlet velocity: `U = Re * nu / D`
- `maxCo = 0.8`
- `maxDeltaT = 0.2 s`
- Parallel subdomains: `8`

## Storage Strategy

Each case is run in a temporary OpenFOAM directory. After solution:

1. `reconstructPar`
2. `foamToVTK -useTimeName -noFaceZones`
3. Convert retained VTK time frames after the transient window into compressed NumPy NPZ.
4. Move logs and case info to result folder.
5. Delete the temporary OpenFOAM case directory.

## Weighted L2 POD

The saved fields are VTK `pointData`. A nodal lumped L2 weight is built from the mesh:

```text
V_i = sum_{cell c containing point i} Volume(c) / N_points(c)
```

Velocity snapshots use:

```text
x = [u_1, ..., u_N, v_1, ..., v_N]
x_weighted = sqrt([V_1, ..., V_N, V_1, ..., V_N]) * x
```

Pressure snapshots use:

```text
y_weighted = sqrt([V_1, ..., V_N]) * y
```

This makes the Euclidean snapshot covariance exactly equal to the corresponding lumped nodal L2 inner product for the stored point fields.
