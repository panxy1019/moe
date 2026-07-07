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
| rhs_l2 | 0.51436 | 0.350417 | 0.162574 | 1.12858 |
| pressure_head_l2 | 0.336654 | 0.276181 | 0.0332621 | 0.924901 |
| one_step_a_l2 | 0.102068 | 0.125256 | 0.017934 | 0.484614 |
| one_step_b_l2 | 133.076 | 177.464 | 0.144666 | 618.334 |
| rollout_a_l2 | 3.65127 | 6.57437 | 0.0849097 | 23.3281 |
| rollout_b_l2 | 223.456 | 284.384 | 0.203548 | 1008.93 |
| one_step_pressure_energy_error | 49124.3 | 108055 | 0.017031 | 381770 |
| rollout_pressure_energy_error | 114553 | 237578 | 0.0455235 | 844913 |

Error curve CSV: `/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_FiLMBase/V15_1_FiLMBase_physics_generalizable_ru16_rp16/V15_1_FiLMBase_physics_generalizable_ru16_rp16_error_vs_re.csv`

Error curve SVG: `/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_FiLMBase/V15_1_FiLMBase_physics_generalizable_ru16_rp16/V15_1_FiLMBase_physics_generalizable_ru16_rp16_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.374731 | 54.866 | - | 0.0219616 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.590098 | 54.866 | 0.6811 | 0.0330798 | 0.0329869 | 289.48 | 0.24761 | 1.15761 | 421.776 | 3 | 3.09255 | 0.692123 | 19 |
| 32.74006652832031 | Galerkin only | 0.329263 | 68.9859 | - | 0.0191824 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.442726 | 68.9859 | 0.444836 | 0.0249253 | 0.024866 | 210.293 | 0.18383 | 1.04695 | 357.287 | 3 | 3.09255 | 0.69212 | 19 |
| 39.68547821044922 | Galerkin only | 0.299669 | 56.0379 | - | 0.0174179 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.36704 | 56.0379 | 0.385958 | 0.0207236 | 0.0206793 | 113.298 | 0.150244 | 0.953853 | 200.17 | 3 | 3.09255 | 0.692121 | 19 |
| 45.142704010009766 | Galerkin only | 0.279522 | 50.0437 | - | 0.0162395 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.31744 | 50.0437 | 0.165792 | 0.017972 | 0.017934 | 74.5744 | 0.121726 | 0.866615 | 129.642 | 3 | 3.09255 | 0.692121 | 19 |
| 47.081356048583984 | Galerkin only | 2.17816 | 66.4526 | - | 0.269108 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 1.10212 | 66.4526 | 0.475237 | 0.109943 | 0.109583 | 90.2665 | 0.458103 | 6.88197 | 195.125 | 3 | 3.09259 | 0.692012 | 19 |
| 49.02235794067383 | Galerkin only | 1.76499 | 53.8859 | - | 0.211056 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 0.911986 | 53.8859 | 0.380234 | 0.0878787 | 0.0875745 | 65.7648 | 0.241243 | 5.11967 | 142.969 | 3 | 3.09259 | 0.692016 | 19 |
| 51.78644943237305 | Galerkin only | 2.70785 | 591.267 | - | 1.57782 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 1.12858 | 591.267 | 0.924901 | 0.485039 | 0.484614 | 618.334 | 1.42096 | 23.3281 | 1008.93 | 3 | 3.09253 | 0.692171 | 19 |
| 70.31463623046875 | Galerkin only | 0.194353 | 1.4395 | - | 0.313395 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.213348 | 1.4395 | 0.131041 | 0.150414 | 0.0903744 | 0.8268 | 0.516384 | 0.409814 | 0.983578 | 4.01 | 1.42648 | 1.05204 | 6 |
| 100.35224914550781 | Galerkin only | 0.255587 | 0.936488 | - | 0.402269 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.219731 | 0.936488 | 0.0338684 | 0.152621 | 0.098546 | 0.567512 | 0.554549 | 0.165017 | 0.701969 | 4.44 | 1.68974 | 1.121 | 10 |
| 149.05923461914062 | Galerkin only | 0.278351 | 0.800668 | - | 0.435167 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.202325 | 0.800668 | 0.0469632 | 0.116624 | 0.0880508 | 0.289346 | 0.354884 | 0.0849097 | 0.203548 | 4.35 | 1.64958 | 1.07285 | 10 |
| 189.86227416992188 | Galerkin only | 0.285972 | 0.715973 | - | 0.44897 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.162574 | 0.715973 | 0.0332621 | 0.11229 | 0.0675367 | 0.144666 | 0.462396 | 0.149397 | 0.233489 | 4.47 | 1.38628 | 1.01085 | 9 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 32.74006652832031 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 39.68547821044922 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 45.142704010009766 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 47.081356048583984 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 49.02235794067383 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 51.78644943237305 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 70.31463623046875 | True | [0.373, 0.500, 0.127] | [0.373, 0.500, 0.127] | 0 |
| 100.35224914550781 | True | [0.158, 0.684, 0.158] | [0.158, 0.684, 0.158] | 0 |
| 149.05923461914062 | True | [0.342, 0.614, 0.044] | [0.342, 0.614, 0.044] | 0 |
| 189.86227416992188 | True | [0.386, 0.456, 0.158] | [0.386, 0.456, 0.158] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.746389 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 32.74006652832031 | 0.743083 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 39.68547821044922 | 0.742478 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 45.142704010009766 | 0.742212 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 47.081356048583984 | 0.757022 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 49.02235794067383 | 0.750739 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 51.78644943237305 | 0.772351 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 70.31463623046875 | 0.727878 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 100.35224914550781 | 0.707921 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 149.05923461914062 | 0.702127 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |
| 189.86227416992188 | 0.610289 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e7 |

Runtime: 41129.83 s.
