# HPRS-MoE-ROM V15 Physics-Generalizable Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

V15 pressure input mode: `pressure_only`. All V15 cases keep the V14 best `[a_t,b_t]` pressure state and differ only by ROM dimension or regime-balanced sampling.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Dense Training Split

Test selection: `regime_default`, time stride=1, Re stride=1.

- Train Re count: 89
- Test Re count: 11
- Excluded Re count from Re sparsity: 0
- Dense train samples before time sparsity: 10970
- Kept train samples: 10970
- Validation samples: 1547
- Test samples: 1350
- Compression vs dense train: 1
- Compression vs all non-test candidates: 0.876408

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 20 | train | 61 | 51 | 51 | 10 | 0 |
| 22.5357 | train | 61 | 51 | 51 | 10 | 0 |
| 24.6304 | test | 61 | 0 | 0 | 0 | 61 |
| 26.6673 | train | 61 | 51 | 51 | 10 | 0 |
| 28.6951 | train | 61 | 51 | 51 | 10 | 0 |
| 30.7204 | train | 61 | 51 | 51 | 10 | 0 |
| 32.7401 | test | 61 | 0 | 0 | 0 | 61 |
| 34.7376 | train | 61 | 51 | 51 | 10 | 0 |
| 36.6578 | train | 60 | 50 | 50 | 10 | 0 |
| 38.3572 | train | 61 | 51 | 51 | 10 | 0 |
| 39.6855 | test | 61 | 0 | 0 | 0 | 61 |
| 40.7115 | train | 61 | 51 | 51 | 10 | 0 |
| 41.5766 | train | 60 | 50 | 50 | 10 | 0 |
| 42.3591 | train | 61 | 51 | 51 | 10 | 0 |
| 43.0939 | train | 61 | 51 | 51 | 10 | 0 |
| 43.7974 | train | 60 | 50 | 50 | 10 | 0 |
| 44.4783 | train | 61 | 51 | 51 | 10 | 0 |
| 45.1427 | test | 61 | 0 | 0 | 0 | 61 |
| 45.7952 | train | 61 | 51 | 51 | 10 | 0 |
| 46.4401 | train | 61 | 51 | 51 | 10 | 0 |
| 47.0814 | test | 158 | 0 | 0 | 0 | 158 |
| 47.7229 | train | 158 | 139 | 139 | 19 | 0 |
| 48.3687 | train | 158 | 139 | 139 | 19 | 0 |
| 49.0224 | test | 158 | 0 | 0 | 0 | 158 |
| 49.6876 | train | 159 | 140 | 140 | 19 | 0 |
| 50.3681 | train | 158 | 139 | 139 | 19 | 0 |
| 51.0668 | train | 158 | 139 | 139 | 19 | 0 |
| 51.7864 | test | 158 | 0 | 0 | 0 | 158 |
| 52.5288 | train | 159 | 140 | 140 | 19 | 0 |
| 53.2942 | train | 158 | 139 | 139 | 19 | 0 |
| 54.0815 | train | 158 | 139 | 139 | 19 | 0 |
| 54.888 | train | 158 | 139 | 139 | 19 | 0 |
| 55.7096 | train | 158 | 139 | 139 | 19 | 0 |
| 56.5433 | train | 158 | 139 | 139 | 19 | 0 |
| 57.39 | train | 159 | 140 | 140 | 19 | 0 |
| 58.2626 | train | 158 | 139 | 139 | 19 | 0 |
| 59.2014 | train | 158 | 139 | 139 | 19 | 0 |
| 60.3077 | train | 159 | 140 | 140 | 19 | 0 |
| 61.756 | train | 158 | 139 | 139 | 19 | 0 |
| 63.4998 | train | 158 | 139 | 139 | 19 | 0 |
| 65.2598 | train | 158 | 139 | 139 | 19 | 0 |
| 66.9701 | train | 158 | 139 | 139 | 19 | 0 |
| 68.6497 | train | 158 | 139 | 139 | 19 | 0 |
| 70.3146 | test | 158 | 0 | 0 | 0 | 158 |
| 71.9729 | train | 158 | 139 | 139 | 19 | 0 |
| 73.6283 | train | 158 | 139 | 139 | 19 | 0 |
| 75.2823 | train | 158 | 139 | 139 | 19 | 0 |
| 76.9358 | train | 158 | 139 | 139 | 19 | 0 |
| 78.589 | train | 158 | 139 | 139 | 19 | 0 |
| 80.242 | train | 158 | 139 | 139 | 19 | 0 |
| 81.8947 | train | 158 | 139 | 139 | 19 | 0 |
| 83.5472 | train | 158 | 139 | 139 | 19 | 0 |
| 85.1991 | train | 158 | 139 | 139 | 19 | 0 |
| 86.8502 | train | 158 | 139 | 139 | 19 | 0 |
| 88.4998 | train | 158 | 139 | 139 | 19 | 0 |
| 90.1473 | train | 158 | 139 | 139 | 19 | 0 |
| 91.7922 | train | 158 | 139 | 139 | 19 | 0 |
| 93.4352 | train | 158 | 139 | 139 | 19 | 0 |
| 95.0817 | train | 158 | 139 | 139 | 19 | 0 |
| 96.7493 | train | 158 | 139 | 139 | 19 | 0 |
| 98.4804 | train | 158 | 139 | 139 | 19 | 0 |
| 100.352 | test | 158 | 0 | 0 | 0 | 158 |
| 102.44 | train | 158 | 139 | 139 | 19 | 0 |
| 104.711 | train | 158 | 139 | 139 | 19 | 0 |
| 107.051 | train | 158 | 139 | 139 | 19 | 0 |
| 109.396 | train | 158 | 139 | 139 | 19 | 0 |
| 111.734 | train | 158 | 139 | 139 | 19 | 0 |
| 114.066 | train | 158 | 139 | 139 | 19 | 0 |
| 116.395 | train | 158 | 139 | 139 | 19 | 0 |
| 118.723 | train | 158 | 139 | 139 | 19 | 0 |
| 121.05 | train | 158 | 139 | 139 | 19 | 0 |
| 123.377 | train | 158 | 139 | 139 | 19 | 0 |
| 125.703 | train | 159 | 140 | 140 | 19 | 0 |
| 128.029 | train | 158 | 139 | 139 | 19 | 0 |
| 130.355 | train | 158 | 139 | 139 | 19 | 0 |
| 132.68 | train | 158 | 139 | 139 | 19 | 0 |
| 135.004 | train | 158 | 139 | 139 | 19 | 0 |
| 137.325 | train | 159 | 140 | 140 | 19 | 0 |
| 139.642 | train | 159 | 140 | 140 | 19 | 0 |
| 141.956 | train | 158 | 139 | 139 | 19 | 0 |
| 144.273 | train | 158 | 139 | 139 | 19 | 0 |
| 146.62 | train | 158 | 139 | 139 | 19 | 0 |
| 149.059 | test | 158 | 0 | 0 | 0 | 158 |
| 151.686 | train | 158 | 139 | 139 | 19 | 0 |
| 154.521 | train | 158 | 139 | 139 | 19 | 0 |
| 157.46 | train | 158 | 139 | 139 | 19 | 0 |
| 160.415 | train | 158 | 139 | 139 | 19 | 0 |
| 163.364 | train | 158 | 139 | 139 | 19 | 0 |
| 166.306 | train | 158 | 139 | 139 | 19 | 0 |
| 169.245 | train | 158 | 139 | 139 | 19 | 0 |
| 172.182 | train | 158 | 139 | 139 | 19 | 0 |
| 175.118 | train | 158 | 139 | 139 | 19 | 0 |
| 178.054 | train | 158 | 139 | 139 | 19 | 0 |
| 180.992 | train | 158 | 139 | 139 | 19 | 0 |
| 183.933 | train | 158 | 139 | 139 | 19 | 0 |
| 186.885 | train | 159 | 140 | 140 | 19 | 0 |
| 189.862 | test | 158 | 0 | 0 | 0 | 158 |
| 192.912 | train | 159 | 140 | 140 | 19 | 0 |
| 196.161 | train | 158 | 139 | 139 | 19 | 0 |
| 200 | train | 159 | 140 | 140 | 19 | 0 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 0.726541 | 0.591635 | 0.160627 | 1.83157 |
| pressure_head_l2 | 0.340482 | 0.381824 | 0.00966539 | 1.35109 |
| one_step_a_l2 | 0.307537 | 0.446256 | 0.040356 | 1.56718 |
| one_step_b_l2 | 68.4135 | 131.577 | 0.190364 | 478.942 |
| rollout_a_l2 | 2.32493 | 2.94045 | 0.301775 | 10.9143 |
| rollout_b_l2 | 95.5585 | 174.501 | 0.674193 | 637.501 |
| one_step_pressure_energy_error | 21983.8 | 65539.3 | 0.14397 | 229211 |
| rollout_pressure_energy_error | 33321.4 | 96511.7 | 0.107109 | 338414 |

Error curve CSV: `/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_RegimeAwareROM/V15_1_RegimeAwareROM_physics_generalizable_ru16_rp16/V15_1_RegimeAwareROM_physics_generalizable_ru16_rp16_error_vs_re.csv`

Error curve SVG: `/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_RegimeAwareROM/V15_1_RegimeAwareROM_physics_generalizable_ru16_rp16/V15_1_RegimeAwareROM_physics_generalizable_ru16_rp16_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.374731 | 54.866 | - | 0.0219616 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.819401 | 54.866 | 0.651977 | 0.182767 | 0.183692 | 38.8775 | 2.73785 | 2.62895 | 53.4026 | 4.52 | 1.90534 | 0.894488 | 16 |
| 32.74006652832031 | Galerkin only | 0.329263 | 68.9859 | - | 0.0191824 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.592024 | 68.9859 | 0.510829 | 0.112521 | 0.113059 | 51.7149 | 1.80535 | 0.943187 | 72.2848 | 4.67 | 1.96753 | 0.933735 | 16 |
| 39.68547821044922 | Galerkin only | 0.299669 | 56.0379 | - | 0.0174179 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.488626 | 56.0379 | 0.281536 | 0.0725202 | 0.0729499 | 44.192 | 1.16043 | 0.454981 | 61.8283 | 4.72 | 2.03356 | 0.944637 | 16 |
| 45.142704010009766 | Galerkin only | 0.279522 | 50.0437 | - | 0.0162395 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.436434 | 50.0437 | 0.230633 | 0.045359 | 0.0454486 | 40.375 | 0.689513 | 0.301775 | 56.3386 | 4.82 | 2.2198 | 0.97207 | 16 |
| 47.081356048583984 | Galerkin only | 2.17816 | 66.4526 | - | 0.269108 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 1.61403 | 66.4526 | 0.433835 | 0.74794 | 0.741578 | 53.5565 | 6.5942 | 3.85664 | 91.2359 | 4.91 | 2.43101 | 0.996456 | 16 |
| 49.02235794067383 | Galerkin only | 1.76499 | 53.8859 | - | 0.211056 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 1.44896 | 53.8859 | 0.212634 | 0.391643 | 0.386789 | 43.5856 | 3.5894 | 2.99723 | 74.7789 | 4.91 | 2.43128 | 0.995658 | 16 |
| 51.78644943237305 | Galerkin only | 2.70785 | 591.267 | - | 1.57782 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 1.83157 | 591.267 | 1.35109 | 1.57062 | 1.56718 | 478.942 | 6.1597 | 10.9143 | 637.501 | 4.48 | 1.93831 | 0.860435 | 16 |
| 70.31463623046875 | Galerkin only | 0.194353 | 1.4395 | - | 0.313395 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.160627 | 1.4395 | 0.0299206 | 0.130401 | 0.110689 | 0.58031 | 1.48779 | 1.41854 | 1.44215 | 4.06 | 2.4205 | 1.09617 | 12 |
| 100.35224914550781 | Galerkin only | 0.255587 | 0.936488 | - | 0.402269 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.21407 | 0.936488 | 0.00966539 | 0.134821 | 0.071629 | 0.311152 | 0.84298 | 0.88745 | 0.948769 | 3.85 | 2.73282 | 0.993583 | 18 |
| 149.05923461914062 | Galerkin only | 0.278351 | 0.800668 | - | 0.435167 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.204558 | 0.800668 | 0.01506 | 0.136659 | 0.0495431 | 0.222729 | 0.340256 | 0.61808 | 0.707801 | 3.92 | 2.72497 | 0.96184 | 18 |
| 189.86227416992188 | Galerkin only | 0.285972 | 0.715973 | - | 0.44897 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.18165 | 0.715973 | 0.0181199 | 0.141203 | 0.040356 | 0.190364 | 0.437375 | 0.553036 | 0.674193 | 4.13 | 2.67701 | 1.0065 | 17 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.525, 0.475, 0.000] | [0.525, 0.475, 0.000] | 0 |
| 32.74006652832031 | True | [0.672, 0.328, 0.000] | [0.672, 0.328, 0.000] | 0 |
| 39.68547821044922 | True | [0.721, 0.279, 0.000] | [0.721, 0.279, 0.000] | 0 |
| 45.142704010009766 | True | [0.820, 0.180, 0.000] | [0.820, 0.180, 0.000] | 0 |
| 47.081356048583984 | True | [0.905, 0.095, 0.000] | [0.905, 0.095, 0.000] | 0 |
| 49.02235794067383 | True | [0.905, 0.095, 0.000] | [0.905, 0.095, 0.000] | 0 |
| 51.78644943237305 | True | [0.481, 0.519, 0.000] | [0.481, 0.519, 0.000] | 0 |
| 70.31463623046875 | True | [0.918, 0.082, 0.000] | [0.918, 0.082, 0.000] | 0 |
| 100.35224914550781 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 149.05923461914062 | True | [0.000, 0.994, 0.006] | [0.000, 0.994, 0.006] | 0 |
| 189.86227416992188 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.991875 | True | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 32.74006652832031 | 0.975933 | True | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 39.68547821044922 | 0.93073 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 45.142704010009766 | 0.879676 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 47.081356048583984 | 0.877877 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 49.02235794067383 | 0.865378 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 51.78644943237305 | 0.850831 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 70.31463623046875 | 0.894868 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 100.35224914550781 | 0.917416 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 149.05923461914062 | 0.954378 | True | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 189.86227416992188 | 0.807312 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 44353.77 s.
