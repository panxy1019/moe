# AttractorMoE-ROM V16 Physics-Generalizable Attractor Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

V16 pressure input mode: `pressure_only`. V16 keeps the V14/V15 best `[a_t,b_t]` pressure state while changing only attractor losses or the lightweight attractor conditioning layer.

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
| rhs_l2 | 0.501507 | 0.396308 | 0.186674 | 1.25117 |
| pressure_head_l2 | 0.246203 | 0.312748 | 0.0173922 | 1.1733 |
| one_step_a_l2 | 0.0776657 | 0.117177 | 0.0129903 | 0.437593 |
| one_step_b_l2 | 0.253686 | 0.309872 | 0.0312057 | 1.17931 |
| rollout_a_l2 | 0.338855 | 0.328728 | 0.0323102 | 1.0107 |
| rollout_b_l2 | 1.14723 | 1.70901 | 0.0457876 | 6.37514 |
| one_step_pressure_energy_error | 0.200939 | 0.322672 | 0.00459387 | 1.17604 |
| rollout_pressure_energy_error | 3.02461 | 7.80431 | 0.00230557 | 27.588 |

Error curve CSV: `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_TrainStableCombined32/V16_1_TrainStableCombined32_ru32_rp32/V16_1_TrainStableCombined32_ru32_rp32_error_vs_re.csv`

Error curve SVG: `/root/moe/V16_1_TrainStableAttractorMoE/test_results_v16_1/results/V16_1_TrainStableCombined32/V16_1_TrainStableCombined32_ru32_rp32/V16_1_TrainStableCombined32_ru32_rp32_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.497642 | 47.4738 | - | 0.0292339 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.462303 | 47.4738 | 0.393978 | 0.0253401 | 0.0255058 | 0.394217 | 0.331882 | 0.331318 | 1.36009 | 4.25 | 2.06244 | 1.0108 | 15 |
| 32.74006652832031 | Galerkin only | 0.422785 | 56.3403 | - | 0.0247627 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.321985 | 56.3403 | 0.224931 | 0.0178609 | 0.017859 | 0.225083 | 0.232266 | 0.232285 | 0.963411 | 4.43 | 1.80684 | 1.01216 | 15 |
| 39.68547821044922 | Galerkin only | 0.385127 | 49.4623 | - | 0.0225734 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.265011 | 49.4623 | 0.179834 | 0.014877 | 0.014866 | 0.179951 | 0.180943 | 0.181007 | 0.742365 | 4.41 | 1.81926 | 1.01204 | 15 |
| 45.142704010009766 | Galerkin only | 0.356736 | 45.695 | - | 0.0209462 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.230703 | 45.695 | 0.186039 | 0.0129942 | 0.0129903 | 0.186117 | 0.142223 | 0.142316 | 0.63797 | 4.41 | 1.81926 | 1.01204 | 15 |
| 47.081356048583984 | Galerkin only | 2.65753 | 67.4831 | - | 0.329896 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 1.25117 | 67.4831 | 0.210531 | 0.101838 | 0.102037 | 0.210623 | 0.905174 | 0.918523 | 1.15244 | 4.11 | 2.39746 | 1.00973 | 15 |
| 49.02235794067383 | Galerkin only | 2.12087 | 54.6819 | - | 0.25518 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 0.949521 | 54.6819 | 0.205338 | 0.0746594 | 0.0748945 | 0.205539 | 0.522177 | 0.558742 | 0.884671 | 4.11 | 2.39746 | 1.00973 | 15 |
| 51.78644943237305 | Galerkin only | 3.13795 | 588.069 | - | 1.86061 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 1.18499 | 588.069 | 1.1733 | 0.437585 | 0.437593 | 1.17931 | 1.19503 | 1.0107 | 6.37514 | 4.14 | 2.31292 | 1.00999 | 15 |
| 70.31463623046875 | Galerkin only | 0.21882 | 1.42014 | - | 0.33963 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.216468 | 1.42014 | 0.0751071 | 0.157223 | 0.0740156 | 0.0968842 | 0.435018 | 0.228571 | 0.3233 | 4.41 | 2.0459 | 1.15039 | 11 |
| 100.35224914550781 | Galerkin only | 0.270851 | 1.01569 | - | 0.410135 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.234712 | 1.01569 | 0.0212614 | 0.165205 | 0.0377568 | 0.0413201 | 0.490925 | 0.0535269 | 0.0806917 | 4.63 | 2.32094 | 1.11868 | 13 |
| 149.05923461914062 | Galerkin only | 0.295979 | 0.830144 | - | 0.443647 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.213037 | 0.830144 | 0.0173922 | 0.135014 | 0.027212 | 0.0312057 | 0.381986 | 0.0381103 | 0.0536812 | 4.6 | 1.84594 | 1.13045 | 8 |
| 189.86227416992188 | Galerkin only | 0.304632 | 0.714768 | - | 0.45917 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.186674 | 0.714768 | 0.0205178 | 0.135104 | 0.029593 | 0.0403053 | 0.435546 | 0.0323102 | 0.0457876 | 4.66 | 1.29118 | 1.10299 | 7 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.246, 0.000, 0.754] | [0.246, 0.000, 0.754] | 0 |
| 32.74006652832031 | True | [0.426, 0.000, 0.574] | [0.426, 0.000, 0.574] | 0 |
| 39.68547821044922 | True | [0.410, 0.000, 0.590] | [0.410, 0.000, 0.590] | 0 |
| 45.142704010009766 | True | [0.410, 0.000, 0.590] | [0.410, 0.000, 0.590] | 0 |
| 47.081356048583984 | True | [0.108, 0.000, 0.892] | [0.108, 0.000, 0.892] | 0 |
| 49.02235794067383 | True | [0.108, 0.000, 0.892] | [0.108, 0.000, 0.892] | 0 |
| 51.78644943237305 | True | [0.139, 0.000, 0.861] | [0.139, 0.000, 0.861] | 0 |
| 70.31463623046875 | True | [0.823, 0.101, 0.076] | [0.823, 0.101, 0.076] | 0 |
| 100.35224914550781 | True | [0.000, 0.930, 0.070] | [0.000, 0.930, 0.070] | 0 |
| 149.05923461914062 | True | [0.127, 0.766, 0.108] | [0.127, 0.766, 0.108] | 0 |
| 189.86227416992188 | True | [0.266, 0.468, 0.266] | [0.266, 0.468, 0.266] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.771544 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 32.74006652832031 | 0.771307 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 39.68547821044922 | 0.770906 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 45.142704010009766 | 0.770511 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 47.081356048583984 | 0.771012 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 49.02235794067383 | 0.771886 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 51.78644943237305 | 0.767614 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 70.31463623046875 | 0.372909 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 100.35224914550781 | 0.728919 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 149.05923461914062 | 0.84696 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 189.86227416992188 | 0.844379 | False | low_Re_lt_80: e14; mid_Re_80_160: e7; high_Re_ge_160: e7 |

Runtime: 42022.91 s.
