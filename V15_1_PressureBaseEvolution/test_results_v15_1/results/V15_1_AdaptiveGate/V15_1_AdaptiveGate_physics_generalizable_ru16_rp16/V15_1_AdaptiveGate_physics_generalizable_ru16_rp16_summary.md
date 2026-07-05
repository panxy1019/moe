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
| rhs_l2 | 0.510298 | 0.4242 | 0.154215 | 1.33414 |
| pressure_head_l2 | 0.205541 | 0.175534 | 0.00617569 | 0.549482 |
| one_step_a_l2 | 0.0898138 | 0.161726 | 0.0127413 | 0.585867 |
| one_step_b_l2 | 0.208665 | 0.172369 | 0.0144103 | 0.550842 |
| rollout_a_l2 | 0.453343 | 0.844993 | 0.01788 | 3.06667 |
| rollout_b_l2 | 0.683393 | 0.71351 | 0.0238318 | 2.51843 |
| one_step_pressure_energy_error | 0.203589 | 0.245338 | 0.000297826 | 0.887993 |
| rollout_pressure_energy_error | 1.2306 | 1.86516 | 0.00379302 | 5.67158 |

Error curve CSV: `/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_AdaptiveGate/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16_error_vs_re.csv`

Error curve SVG: `/root/moe/V15_1_PressureBaseEvolution/test_results_v15_1/results/V15_1_AdaptiveGate/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16/V15_1_AdaptiveGate_physics_generalizable_ru16_rp16_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.374731 | 54.866 | - | 0.0219616 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.455586 | 54.866 | 0.454716 | 0.0257217 | 0.0257337 | 0.454299 | 0.241417 | 0.243583 | 1.39995 | 4.66 | 2.02911 | 1.01353 | 15 |
| 32.74006652832031 | Galerkin only | 0.329263 | 68.9859 | - | 0.0191824 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.341855 | 68.9859 | 0.305619 | 0.0193495 | 0.0193662 | 0.305318 | 0.181713 | 0.183362 | 0.947332 | 4.7 | 2.42266 | 1.01146 | 15 |
| 39.68547821044922 | Galerkin only | 0.299669 | 56.0379 | - | 0.0174179 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.282107 | 56.0379 | 0.232853 | 0.0159964 | 0.0160287 | 0.232686 | 0.152738 | 0.15462 | 0.717492 | 4.7 | 2.71122 | 1.00986 | 18 |
| 45.142704010009766 | Galerkin only | 0.279522 | 50.0437 | - | 0.0162395 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.239249 | 50.0437 | 0.234676 | 0.0135978 | 0.0136368 | 0.234489 | 0.126498 | 0.126533 | 0.621411 | 4.7 | 2.71121 | 1.0099 | 18 |
| 47.081356048583984 | Galerkin only | 2.17816 | 66.4526 | - | 0.269108 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 1.33414 | 66.4526 | 0.207575 | 0.138525 | 0.138868 | 0.207886 | 0.677501 | 0.619731 | 0.562771 | 5 | 2.24808 | 1.01262 | 15 |
| 49.02235794067383 | Galerkin only | 1.76499 | 53.8859 | - | 0.211056 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 0.987499 | 53.8859 | 0.219305 | 0.100147 | 0.100466 | 0.219253 | 0.405456 | 0.427682 | 0.552757 | 5 | 2.12806 | 1.01356 | 15 |
| 51.78644943237305 | Galerkin only | 2.70785 | 591.267 | - | 1.57782 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 1.21769 | 591.267 | 0.549482 | 0.585691 | 0.585867 | 0.550842 | 2.8014 | 3.06667 | 2.51843 | 5 | 1.78577 | 1.01998 | 15 |
| 70.31463623046875 | Galerkin only | 0.194353 | 1.4395 | - | 0.313395 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.154215 | 1.4395 | 0.0271659 | 0.147209 | 0.0390687 | 0.0353594 | 0.696077 | 0.094213 | 0.108112 | 4.66 | 2.47521 | 1.13379 | 15 |
| 100.35224914550781 | Galerkin only | 0.255587 | 0.936488 | - | 0.402269 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.214549 | 0.936488 | 0.00806922 | 0.15462 | 0.0179397 | 0.0159687 | 0.538637 | 0.0266568 | 0.0301158 | 4.65 | 2.50258 | 1.17857 | 15 |
| 149.05923461914062 | Galerkin only | 0.278351 | 0.800668 | - | 0.435167 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.205088 | 0.800668 | 0.00617569 | 0.165649 | 0.0127413 | 0.0144103 | 0.714171 | 0.01788 | 0.0238318 | 4.72 | 2.517 | 1.19876 | 15 |
| 189.86227416992188 | Galerkin only | 0.285972 | 0.715973 | - | 0.44897 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.181306 | 0.715973 | 0.0153137 | 0.179948 | 0.0182352 | 0.0248106 | 0.762459 | 0.0258505 | 0.0351285 | 3.81 | 2.61611 | 0.888347 | 14 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.738, 0.262, 0.000] | [0.738, 0.262, 0.000] | 0 |
| 32.74006652832031 | True | [0.902, 0.098, 0.000] | [0.902, 0.098, 0.000] | 0 |
| 39.68547821044922 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 45.142704010009766 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 47.081356048583984 | True | [0.835, 0.165, 0.000] | [0.835, 0.165, 0.000] | 0 |
| 49.02235794067383 | True | [0.785, 0.215, 0.000] | [0.785, 0.215, 0.000] | 0 |
| 51.78644943237305 | True | [0.462, 0.538, 0.000] | [0.462, 0.538, 0.000] | 0 |
| 70.31463623046875 | True | [0.968, 0.032, 0.000] | [0.968, 0.032, 0.000] | 0 |
| 100.35224914550781 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 149.05923461914062 | True | [0.000, 0.994, 0.006] | [0.000, 0.994, 0.006] | 0 |
| 189.86227416992188 | True | [0.070, 0.013, 0.918] | [0.070, 0.013, 0.918] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.550668 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 32.74006652832031 | 0.553584 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 39.68547821044922 | 0.555474 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 45.142704010009766 | 0.556726 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 47.081356048583984 | 0.559206 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 49.02235794067383 | 0.559544 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 51.78644943237305 | 0.560703 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 70.31463623046875 | 0.82612 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 100.35224914550781 | 0.876447 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 149.05923461914062 | 0.944619 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 189.86227416992188 | 0.900925 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 39719.32 s.
