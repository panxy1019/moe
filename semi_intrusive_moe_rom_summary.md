# Semi-intrusive Galerkin + Shared-Routed MoE Test Summary

## Scheme

Reduced RHS: `adot = R_galerkin(a,b;Re) + C_shared(x) + sum_e gate_e(Re,phase) C_e(x)`.

Ranks: velocity r_u=24, pressure r_p=24; phase Fourier harmonics K=4; top-k router=2.

## Metrics

| Test Re | Model | RHS relative L2 | RHS RMSE | centered R2 | one-step relative L2 | improvement vs Galerkin |
|---:|---|---:|---:|---:|---:|---:|
| 700 | Galerkin only | 0.197367 | 1.08982 | 0.961006 | 0.0677599 | 0% |
| 700 | Galerkin + shared | 0.178188 | 0.983919 | 0.968216 | 0.0609714 | 9.71749% |
| 700 | Galerkin + shared-routed | 0.165008 | 0.911139 | 0.972744 | 0.0594975 | 16.3957% |
| 1000 | Galerkin only | 0.17517 | 2.21326 | 0.969282 | 0.0644952 | 0% |
| 1000 | Galerkin + shared | 0.149011 | 1.88275 | 0.977772 | 0.0555187 | 14.9333% |
| 1000 | Galerkin + shared-routed | 0.141604 | 1.78916 | 0.979927 | 0.0540458 | 19.1617% |

## Rollout

Rollout is a lightweight Euler check using true pressure coefficients and known phase as context; it is not yet a fully autonomous pressure-coupled ROM.

| Test Re | steps | windows | mean relative L2 | median relative L2 |
|---:|---:|---:|---:|---:|
| 700 | 20 | 9 | 0.637304 | 0.475193 |
| 1000 | 20 | 9 | 0.502369 | 0.383603 |

Runtime: 2.00035 s.
