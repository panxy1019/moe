# Semi-intrusive Galerkin + Shared-Routed MoE Test Summary

## Scheme

Reduced RHS: `adot = R_galerkin(a,b;Re) + C_shared(x) + sum_e gate_e(Re,phase) C_e(x)`.

Ranks: velocity r_u=16, pressure r_p=16; phase Fourier harmonics K=4; top-k router=2.

## Metrics

| Test Re | Model | RHS relative L2 | RHS RMSE | centered R2 | one-step relative L2 | improvement vs Galerkin |
|---:|---|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.203423 | 1.29909 | 0.958575 | 0.0649938 | 0% |
| 700 | Galerkin + shared | 0.142434 | 0.909605 | 0.979691 | 0.0518297 | 29.9812% |
| 700 | Galerkin + shared-routed | 0.11752 | 0.750502 | 0.986174 | 0.0490176 | 42.2285% |
| 1000 | Galerkin only | 0.17964 | 2.6541 | 0.967693 | 0.057334 | 0% |
| 1000 | Galerkin + shared | 0.10774 | 1.5918 | 0.988379 | 0.0474805 | 40.0248% |
| 1000 | Galerkin + shared-routed | 0.0985543 | 1.45609 | 0.990276 | 0.045681 | 45.138% |

## Rollout

Rollout is a lightweight Euler check using true pressure coefficients and known phase as context; it is not yet a fully autonomous pressure-coupled ROM.

| Test Re | steps | windows | mean relative L2 | median relative L2 |
|---:|---:|---:|---:|---:|
| 700 | 20 | 9 | 0.426427 | 0.33093 |
| 1000 | 20 | 9 | 0.400615 | 0.385423 |

Runtime: 1.87741 s.
