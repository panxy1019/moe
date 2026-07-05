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
| rhs_l2 | 0.889033 | 0.911777 | 0.153841 | 2.86914 |
| pressure_head_l2 | 0.551077 | 0.608397 | 0.0192711 | 1.87833 |
| one_step_a_l2 | 0.184056 | 0.383905 | 0.0234602 | 1.38032 |
| one_step_b_l2 | 0.594782 | 0.661085 | 0.0386793 | 1.87189 |
| rollout_a_l2 | 0.473036 | 0.702239 | 0.0450402 | 2.55039 |
| rollout_b_l2 | 1.05981 | 1.62775 | 0.0559842 | 6.03049 |
| one_step_pressure_energy_error | 0.937857 | 1.36614 | 0.0032464 | 4.05525 |
| rollout_pressure_energy_error | 3.34869 | 9.29536 | 0.00451132 | 32.6985 |

Error curve CSV: `/root/moe/V15_PhysicsGeneralizable/test_results_v15/results/V15_BalancedTraining/V15_BalancedTraining_physics_generalizable_ru16_rp16/V15_BalancedTraining_physics_generalizable_ru16_rp16_error_vs_re.csv`

Error curve SVG: `/root/moe/V15_PhysicsGeneralizable/test_results_v15/results/V15_BalancedTraining/V15_BalancedTraining_physics_generalizable_ru16_rp16/V15_BalancedTraining_physics_generalizable_ru16_rp16_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.374731 | 54.866 | - | 0.0219616 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.641247 | 54.866 | 1.87833 | 0.0360066 | 0.0359596 | 1.87189 | 0.257775 | 0.256882 | 0.813858 | 4.2 | 2.17543 | 1.00722 | 15 |
| 32.74006652832031 | Galerkin only | 0.329263 | 68.9859 | - | 0.0191824 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.48989 | 68.9859 | 1.03181 | 0.0276785 | 0.0281497 | 1.01876 | 0.212925 | 0.22312 | 1.0883 | 4.23 | 2.04823 | 1.00727 | 12 |
| 39.68547821044922 | Galerkin only | 0.299669 | 56.0379 | - | 0.0174179 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.474178 | 56.0379 | 0.29424 | 0.0266375 | 0.0270027 | 0.295546 | 0.192001 | 0.196136 | 0.592634 | 4.49 | 1.67858 | 1.00787 | 12 |
| 45.142704010009766 | Galerkin only | 0.279522 | 50.0437 | - | 0.0162395 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.469349 | 50.0437 | 0.278376 | 0.0260711 | 0.0268604 | 0.287119 | 0.227273 | 0.228361 | 0.561414 | 4.9 | 2.42299 | 1.00846 | 15 |
| 47.081356048583984 | Galerkin only | 2.17816 | 66.4526 | - | 0.269108 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 2.20123 | 66.4526 | 0.496942 | 0.217508 | 0.217623 | 0.524043 | 0.876561 | 0.850623 | 1.28989 | 4.37 | 1.79855 | 1.008 | 14 |
| 49.02235794067383 | Galerkin only | 1.76499 | 53.8859 | - | 0.211056 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 1.87592 | 53.8859 | 0.467255 | 0.182529 | 0.182861 | 0.484204 | 0.676541 | 0.660197 | 0.963075 | 4.45 | 1.72074 | 1.0081 | 14 |
| 51.78644943237305 | Galerkin only | 2.70785 | 591.267 | - | 1.57782 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 2.86914 | 591.267 | 1.47799 | 1.37978 | 1.38032 | 1.86214 | 3.2419 | 2.55039 | 6.03049 | 4.73 | 1.85697 | 1.00864 | 12 |
| 70.31463623046875 | Galerkin only | 0.194353 | 1.4395 | - | 0.313395 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.153841 | 1.4395 | 0.0587783 | 0.132482 | 0.0430593 | 0.0705026 | 0.58278 | 0.0927259 | 0.13328 | 4.23 | 1.60362 | 1.07559 | 9 |
| 100.35224914550781 | Galerkin only | 0.255587 | 0.936488 | - | 0.402269 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.216929 | 0.936488 | 0.0256243 | 0.166126 | 0.0356975 | 0.0492666 | 0.472941 | 0.0475451 | 0.0623082 | 4.57 | 2.43946 | 1.12678 | 14 |
| 149.05923461914062 | Galerkin only | 0.278351 | 0.800668 | - | 0.435167 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.205402 | 0.800668 | 0.0192711 | 0.176518 | 0.0236213 | 0.0386793 | 0.509931 | 0.0450402 | 0.0559842 | 4.26 | 1.96948 | 1.10739 | 11 |
| 189.86227416992188 | Galerkin only | 0.285972 | 0.715973 | - | 0.44897 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.18224 | 0.715973 | 0.0332248 | 0.176607 | 0.0234602 | 0.0404509 | 0.606272 | 0.0523728 | 0.0666508 | 4.09 | 1.72832 | 1.07562 | 11 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.000, 0.197, 0.803] | [0.000, 0.197, 0.803] | 0 |
| 32.74006652832031 | True | [0.131, 0.098, 0.770] | [0.131, 0.098, 0.770] | 0 |
| 39.68547821044922 | True | [0.443, 0.049, 0.508] | [0.443, 0.049, 0.508] | 0 |
| 45.142704010009766 | True | [0.902, 0.000, 0.098] | [0.902, 0.000, 0.098] | 0 |
| 47.081356048583984 | True | [0.335, 0.038, 0.627] | [0.335, 0.038, 0.627] | 0 |
| 49.02235794067383 | True | [0.411, 0.038, 0.551] | [0.411, 0.038, 0.551] | 0 |
| 51.78644943237305 | True | [0.677, 0.057, 0.266] | [0.677, 0.057, 0.266] | 0 |
| 70.31463623046875 | True | [0.456, 0.481, 0.063] | [0.456, 0.481, 0.063] | 0 |
| 100.35224914550781 | True | [0.000, 0.968, 0.032] | [0.000, 0.968, 0.032] | 0 |
| 149.05923461914062 | True | [0.013, 0.797, 0.190] | [0.013, 0.797, 0.190] | 0 |
| 189.86227416992188 | True | [0.000, 0.620, 0.380] | [0.000, 0.620, 0.380] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.714787 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 32.74006652832031 | 0.700913 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 39.68547821044922 | 0.700328 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 45.142704010009766 | 0.705195 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 47.081356048583984 | 0.700028 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 49.02235794067383 | 0.701169 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 51.78644943237305 | 0.707898 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 70.31463623046875 | 0.814593 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 100.35224914550781 | 0.828304 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 149.05923461914062 | 0.799721 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 189.86227416992188 | 0.864535 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |

Runtime: 37530.31 s.
