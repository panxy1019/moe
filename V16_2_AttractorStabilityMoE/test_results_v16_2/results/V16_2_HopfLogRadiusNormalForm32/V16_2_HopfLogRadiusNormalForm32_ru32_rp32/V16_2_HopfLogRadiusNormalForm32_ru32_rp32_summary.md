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
| rhs_l2 | 2.09964 | 1.86693 | 0.277902 | 5.6595 |
| pressure_head_l2 | 0.800537 | 0.713494 | 0.108924 | 2.65713 |
| one_step_a_l2 | 0.344264 | 0.565623 | 0.0588599 | 2.06527 |
| one_step_b_l2 | 0.852659 | 0.81959 | 0.118503 | 3.09303 |
| rollout_a_l2 | 1.08943 | 1.29547 | 0.226001 | 4.96388 |
| rollout_b_l2 | 2.19119 | 2.17942 | 0.490599 | 8.41116 |
| one_step_pressure_energy_error | 0.93295 | 1.73881 | 0.0666032 | 6.37046 |
| rollout_pressure_energy_error | 7.32579 | 14.7368 | 0.131231 | 52.0396 |

Error curve CSV: `/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2/results/V16_2_HopfLogRadiusNormalForm32/V16_2_HopfLogRadiusNormalForm32_ru32_rp32/V16_2_HopfLogRadiusNormalForm32_ru32_rp32_error_vs_re.csv`

Error curve SVG: `/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2/results/V16_2_HopfLogRadiusNormalForm32/V16_2_HopfLogRadiusNormalForm32_ru32_rp32/V16_2_HopfLogRadiusNormalForm32_ru32_rp32_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.497642 | 47.4738 | - | 0.0292339 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 2.66029 | 47.4738 | 1.01367 | 0.137601 | 0.137362 | 1.00528 | 1.064 | 1.04515 | 3.29508 | 5 | 2.71099 | 1.01089 | 18 |
| 32.74006652832031 | Galerkin only | 0.422785 | 56.3403 | - | 0.0247627 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 1.87726 | 56.3403 | 0.571042 | 0.0991552 | 0.0989681 | 0.562097 | 0.685077 | 0.662077 | 2.68234 | 5 | 2.71099 | 1.01087 | 18 |
| 39.68547821044922 | Galerkin only | 0.385127 | 49.4623 | - | 0.0225734 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 1.49188 | 49.4623 | 0.362619 | 0.0799133 | 0.0798071 | 0.367113 | 0.512053 | 0.497362 | 1.69805 | 5 | 2.71099 | 1.01088 | 18 |
| 45.142704010009766 | Galerkin only | 0.356736 | 45.695 | - | 0.0209462 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 1.27404 | 45.695 | 0.603264 | 0.0689404 | 0.0688798 | 0.609028 | 0.419159 | 0.412577 | 1.09916 | 5 | 2.71099 | 1.01089 | 18 |
| 47.081356048583984 | Galerkin only | 2.65753 | 67.4831 | - | 0.329896 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 5.6595 | 67.4831 | 1.30669 | 0.533931 | 0.533958 | 1.35144 | 1.75833 | 1.67503 | 2.75089 | 5 | 2.71099 | 1.01086 | 18 |
| 49.02235794067383 | Galerkin only | 2.12087 | 54.6819 | - | 0.25518 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 5.00412 | 54.6819 | 1.29152 | 0.442535 | 0.442753 | 1.33125 | 1.24623 | 1.21792 | 1.75444 | 5 | 2.71099 | 1.01087 | 18 |
| 51.78644943237305 | Galerkin only | 3.13795 | 588.069 | - | 1.86061 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 3.85983 | 588.069 | 2.65713 | 2.06575 | 2.06527 | 3.09303 | 5.13722 | 4.96388 | 8.41116 | 5 | 2.71098 | 1.01093 | 18 |
| 70.31463623046875 | Galerkin only | 0.21882 | 1.42014 | - | 0.33963 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.406416 | 1.42014 | 0.52654 | 0.146421 | 0.143918 | 0.532405 | 0.617806 | 0.557774 | 0.825026 | 4.59 | 1.71671 | 1.05047 | 12 |
| 100.35224914550781 | Galerkin only | 0.270851 | 1.01569 | - | 0.410135 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.304144 | 1.01569 | 0.24886 | 0.090612 | 0.0872259 | 0.265532 | 0.441948 | 0.443524 | 0.589402 | 4.54 | 1.56029 | 1.11591 | 10 |
| 149.05923461914062 | Galerkin only | 0.295979 | 0.830144 | - | 0.443647 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.277902 | 0.830144 | 0.108924 | 0.0698555 | 0.0699022 | 0.143566 | 0.288784 | 0.282434 | 0.506928 | 4.27 | 1.23763 | 1.19841 | 4 |
| 189.86227416992188 | Galerkin only | 0.304632 | 0.714768 | - | 0.45917 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.280699 | 0.714768 | 0.115645 | 0.0593628 | 0.0588599 | 0.118503 | 0.239258 | 0.226001 | 0.490599 | 4.32 | 1.21584 | 1.19743 | 4 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 32.74006652832031 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 39.68547821044922 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 45.142704010009766 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 47.081356048583984 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 49.02235794067383 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 51.78644943237305 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 70.31463623046875 | True | [0.215, 0.639, 0.146] | [0.215, 0.639, 0.146] | 0 |
| 100.35224914550781 | True | [0.222, 0.582, 0.196] | [0.222, 0.582, 0.196] | 0 |
| 149.05923461914062 | True | [0.399, 0.234, 0.367] | [0.399, 0.234, 0.367] | 0 |
| 189.86227416992188 | True | [0.291, 0.424, 0.285] | [0.291, 0.424, 0.285] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.506091 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 32.74006652832031 | 0.507249 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 39.68547821044922 | 0.508143 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 45.142704010009766 | 0.508681 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 47.081356048583984 | 0.506662 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 49.02235794067383 | 0.5072 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 51.78644943237305 | 0.50701 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 70.31463623046875 | 0.378539 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 100.35224914550781 | 0.348755 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 149.05923461914062 | 0.318335 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 189.86227416992188 | 0.243114 | False | low_Re_lt_80: e7; mid_Re_80_160: e7; high_Re_ge_160: e0 |

Runtime: 24465.97 s.
