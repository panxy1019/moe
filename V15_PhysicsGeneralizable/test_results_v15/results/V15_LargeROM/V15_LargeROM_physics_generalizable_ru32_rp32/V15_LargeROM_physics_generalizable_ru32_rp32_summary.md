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
| rhs_l2 | 0.635628 | 0.436371 | 0.185701 | 1.37919 |
| pressure_head_l2 | 0.573006 | 0.581195 | 0.014197 | 1.64829 |
| one_step_a_l2 | 0.102827 | 0.167403 | 0.0218651 | 0.620734 |
| one_step_b_l2 | 0.603573 | 0.612555 | 0.0406584 | 1.77965 |
| rollout_a_l2 | 0.453143 | 0.697082 | 0.0293711 | 2.5567 |
| rollout_b_l2 | 1.4827 | 2.64779 | 0.0431974 | 9.63969 |
| one_step_pressure_energy_error | 0.724721 | 1.10194 | 0.000146054 | 3.01705 |
| rollout_pressure_energy_error | 7.81779 | 23.0708 | 0.00558528 | 80.7429 |

Error curve CSV: `/root/moe/V15_PhysicsGeneralizable/test_results_v15/results/V15_LargeROM/V15_LargeROM_physics_generalizable_ru32_rp32/V15_LargeROM_physics_generalizable_ru32_rp32_error_vs_re.csv`

Error curve SVG: `/root/moe/V15_PhysicsGeneralizable/test_results_v15/results/V15_LargeROM/V15_LargeROM_physics_generalizable_ru32_rp32/V15_LargeROM_physics_generalizable_ru32_rp32_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.497642 | 47.4738 | - | 0.0292339 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.846176 | 47.4738 | 1.64829 | 0.045584 | 0.045087 | 1.67285 | 0.432817 | 0.425509 | 1.5048 | 5 | 1.78958 | 1.01155 | 15 |
| 32.74006652832031 | Galerkin only | 0.422785 | 56.3403 | - | 0.0247627 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.622505 | 56.3403 | 0.444145 | 0.0341583 | 0.0338526 | 0.442431 | 0.252535 | 0.251023 | 1.69548 | 5 | 1.5761 | 1.01243 | 12 |
| 39.68547821044922 | Galerkin only | 0.385127 | 49.4623 | - | 0.0225734 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.506296 | 49.4623 | 0.357021 | 0.0277302 | 0.0273679 | 0.34003 | 0.181361 | 0.181177 | 0.637421 | 5 | 1.55576 | 1.0125 | 12 |
| 45.142704010009766 | Galerkin only | 0.356736 | 45.695 | - | 0.0209462 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.401626 | 45.695 | 0.43877 | 0.0219889 | 0.0218651 | 0.402754 | 0.133008 | 0.135052 | 0.321955 | 5 | 1.5762 | 1.01124 | 12 |
| 47.081356048583984 | Galerkin only | 2.65753 | 67.4831 | - | 0.329896 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 1.37919 | 67.4831 | 1.1349 | 0.136101 | 0.136095 | 1.13052 | 0.694714 | 0.716017 | 1.47439 | 4.87 | 1.95704 | 1.00989 | 15 |
| 49.02235794067383 | Galerkin only | 2.12087 | 54.6819 | - | 0.25518 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 1.05624 | 54.6819 | 0.632239 | 0.101189 | 0.101101 | 0.633898 | 0.477851 | 0.488784 | 0.720106 | 4.68 | 2.10917 | 1.00951 | 15 |
| 51.78644943237305 | Galerkin only | 3.13795 | 588.069 | - | 1.86061 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 1.35187 | 588.069 | 1.55001 | 0.620696 | 0.620734 | 1.77965 | 1.97474 | 2.5567 | 9.63969 | 4.22 | 2.12858 | 1.00957 | 15 |
| 70.31463623046875 | Galerkin only | 0.21882 | 1.42014 | - | 0.33963 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.201783 | 1.42014 | 0.0445836 | 0.140352 | 0.0607033 | 0.0923773 | 0.55494 | 0.130786 | 0.169433 | 4.28 | 2.44333 | 1.05937 | 15 |
| 100.35224914550781 | Galerkin only | 0.270851 | 1.01569 | - | 0.410135 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.227426 | 1.01569 | 0.0152953 | 0.145152 | 0.0327494 | 0.0457977 | 0.680175 | 0.033108 | 0.0512348 | 4.2 | 2.16623 | 1.10587 | 10 |
| 149.05923461914062 | Galerkin only | 0.295979 | 0.830144 | - | 0.443647 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.213092 | 0.830144 | 0.014197 | 0.12055 | 0.0222699 | 0.0406584 | 0.402335 | 0.0293711 | 0.0431974 | 4.01 | 2.48684 | 0.975138 | 14 |
| 189.86227416992188 | Galerkin only | 0.304632 | 0.714768 | - | 0.45917 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.185701 | 0.714768 | 0.0236109 | 0.140886 | 0.0292731 | 0.0583293 | 0.593718 | 0.0370411 | 0.0520393 | 3.95 | 2.02295 | 0.912874 | 11 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.016, 0.393, 0.590] | [0.016, 0.393, 0.590] | 0 |
| 32.74006652832031 | True | [0.328, 0.131, 0.541] | [0.328, 0.131, 0.541] | 0 |
| 39.68547821044922 | True | [0.492, 0.115, 0.393] | [0.492, 0.115, 0.393] | 0 |
| 45.142704010009766 | True | [0.426, 0.098, 0.475] | [0.426, 0.098, 0.475] | 0 |
| 47.081356048583984 | True | [0.291, 0.006, 0.703] | [0.291, 0.006, 0.703] | 0 |
| 49.02235794067383 | True | [0.215, 0.006, 0.778] | [0.215, 0.006, 0.778] | 0 |
| 51.78644943237305 | True | [0.215, 0.000, 0.785] | [0.215, 0.000, 0.785] | 0 |
| 70.31463623046875 | True | [0.956, 0.038, 0.006] | [0.956, 0.038, 0.006] | 0 |
| 100.35224914550781 | True | [0.032, 0.861, 0.108] | [0.032, 0.861, 0.108] | 0 |
| 149.05923461914062 | True | [0.000, 0.949, 0.051] | [0.000, 0.949, 0.051] | 0 |
| 189.86227416992188 | True | [0.000, 0.759, 0.241] | [0.000, 0.759, 0.241] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.793773 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 32.74006652832031 | 0.797159 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 39.68547821044922 | 0.802357 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 45.142704010009766 | 0.809642 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 47.081356048583984 | 0.815864 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 49.02235794067383 | 0.820817 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 51.78644943237305 | 0.82654 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 70.31463623046875 | 0.602826 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 100.35224914550781 | 0.740756 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 149.05923461914062 | 0.654597 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 189.86227416992188 | 0.788169 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |

Runtime: 37399.15 s.
