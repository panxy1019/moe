# Re 50-300 Cylinder Dataset And Weighted L2 POD Operations

Generated on: 2026-06-17

## Physical Scope

- Circular cylinder diameter: `D = 1 m`.
- Kinematic viscosity: `nu = 1e-3 m^2/s`.
- Reynolds number range: `Re = 50-300`, `100` nonuniform samples.
- Sampling strategy:
  - `Re = 50-120`: dense nonuniform sampling near the onset of vortex shedding.
  - `Re = 120-300`: moderately sparse nonuniform sampling.
- The circular-cylinder wake begins periodic vortex shedding near `Re ~ 47`.
- The `Re > ~188` part of this dataset should be treated as a two-dimensional constrained model for ROM/ML basis construction, not as a quantitatively faithful real 3D cylinder wake. Published Floquet analyses report 3D wake instability around `Re = 188.5 +/- 1.0`, with a second instability branch around `Re = 259`.

References:

- Williamson, C. H. K. (1996), "Vortex Dynamics in the Cylinder Wake", Annual Review of Fluid Mechanics. DOI: https://doi.org/10.1146/annurev.fl.28.010196.002401
- Barkley & Henderson (1996), "Three-dimensional Floquet stability analysis of the wake of a circular cylinder", Journal of Fluid Mechanics. https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/threedimensional-floquet-stability-analysis-of-the-wake-of-a-circular-cylinder/61575FBF0BC45054592D46382DEF30BB
- Noack & Eckelmann (1994), "A global stability analysis of the steady and periodic cylinder wake", Journal of Fluid Mechanics. https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/global-stability-analysis-of-the-steady-and-periodic-cylinder-wake/53544020CECA810BA65AC737FF23B381

## Simulation Setup

- Solver: `pimpleFoam`.
- Mesh: `blockMesh`, 2D empty front/back patches.
- Mesh size from `blockMesh`:
  - points: `97368`
  - cells: `48128`
  - domain: `(-10, -10, -0.05)` to `(20, 10, 0.05)`
- Transient handling:
  - total simulated duration per Re: estimated `20` shedding periods.
  - first `4` periods dropped.
  - retained duration: about `16` shedding periods.
  - retained snapshots: `128-129` per Re.
- Time control:
  - adaptive time step
  - target `maxCo = 0.8`
  - VTK was converted immediately to compressed `npz`; temporary OpenFOAM run directories were removed after each successful case.

## Dataset Files

Root directory:

`/home/ray/Desktop/Cylinder_Results_Re50_300_100Re_POD`

Per-Re files:

- `Re_*_uvp_pointData.npz`: compressed point-field data containing `u`, `v`, `p`, coordinates, time array, and metadata.
- `Re_*_uvp_dimensions.txt`: human-readable data dimensions for each Re.
- `Re_*_uvp_metadata.json`: per-case metadata.
- `Re_*_info.txt`: run settings and summary.
- `Re_*_logs/`: OpenFOAM logs.

Global summaries:

- `Simulation_Summary.csv`
- `uvp_npz_summary.json`
- `RUN_COMPLETE.txt`

## Weighted L2 POD

Output directory:

`/home/ray/Desktop/Cylinder_Results_Re50_300_100Re_POD/Global_POD_Weighted_L2`

The POD was assembled with mesh-volume weighting to prevent dense near-cylinder grid regions from dominating the Euclidean SVD. The mesh cell volumes were computed from the OpenFOAM mesh exported to VTK. Because stored fields are point data, each cell volume was distributed equally to its cell vertices to form a lumped nodal control-volume vector:

`V_point[i] = sum(cell volumes adjacent to point i / number of vertices in cell)`

The weighted matrices used for SVD were:

- velocity: `X_uv_weighted = [sqrt(V) * u_fluct, sqrt(V) * v_fluct]`
- pressure: `X_p_weighted = sqrt(V) * p_fluct`

This makes the snapshot dot products equivalent to a lumped discrete `L2` inner product:

- velocity: `<q_i, q_j>_L2 ~= sum_k V_k * (u_i,k u_j,k + v_i,k v_j,k)`
- pressure: `<p_i, p_j>_L2 ~= sum_k V_k * p_i,k p_j,k`

Weighted POD outputs:

- `mesh_l2_point_weights.npz`
- `global_velocity_pod_weighted_l2.npz`
- `global_pressure_pod_weighted_l2.npz`
- `pod_snapshot_index.csv`
- `pod_weighted_l2_metadata.json`

Verified dimensions:

- total snapshots: `12869`
- point weights: `(97368,)`
- velocity modes `phi_uv`: `(80, 194736)`
- velocity coefficients `coeff_uv`: `(12869, 80)`
- pressure modes `phi_p`: `(80, 97368)`
- pressure coefficients `coeff_p`: `(12869, 80)`
- retained velocity weighted energy at 80 modes: `0.9970816818416127`
- retained pressure weighted energy at 80 modes: `0.9987816277255009`

## Execution Notes

- The run was interrupted once during `Re_187p285235`. The partial temporary case was detected and the main script was restarted.
- The automation skipped the existing `76` complete cases and resumed from `Re_187p285235`; no completed cases were recomputed.
- The first POD attempt failed during velocity modal back-substitution because NumPy attempted to allocate a large float64 intermediate array.
- The POD implementation was patched to perform modal back-substitution in column blocks using float32 working arrays. OpenFOAM simulations were not rerun for this fix.
- Final disk status after completion: about `12.93 GiB` free.

