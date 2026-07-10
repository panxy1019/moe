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
| rhs_l2 | 0.359727 | 0.186764 | 0.192536 | 0.719227 |
| pressure_head_l2 | 0.114548 | 0.0957717 | 0.014536 | 0.344604 |
| one_step_a_l2 | 0.0532435 | 0.0715825 | 0.012062 | 0.272887 |
| one_step_b_l2 | 0.120451 | 0.0905364 | 0.0283156 | 0.343736 |
| rollout_a_l2 | 0.262538 | 0.352509 | 0.0280086 | 1.32382 |
| rollout_b_l2 | 0.451238 | 0.601325 | 0.0387218 | 2.20781 |
| one_step_pressure_energy_error | 0.0758249 | 0.0676433 | 0.000445627 | 0.178923 |
| rollout_pressure_energy_error | 0.683004 | 1.72706 | 0.000314331 | 6.12631 |

Error curve CSV: `/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2/results/V16_2_SteadyContractivePressureROM32/V16_2_SteadyContractivePressureROM32_ru32_rp32/V16_2_SteadyContractivePressureROM32_ru32_rp32_error_vs_re.csv`

Error curve SVG: `/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2/results/V16_2_SteadyContractivePressureROM32/V16_2_SteadyContractivePressureROM32_ru32_rp32/V16_2_SteadyContractivePressureROM32_ru32_rp32_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.497642 | 47.4738 | - | 0.0292339 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.450076 | 47.4738 | 0.239423 | 0.0246728 | 0.0246491 | 0.239321 | 0.299561 | 0.299352 | 0.67483 | 4 | 2.5241 | 0.748356 | 16 |
| 32.74006652832031 | Galerkin only | 0.422785 | 56.3403 | - | 0.0247627 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.289304 | 56.3403 | 0.125198 | 0.0160875 | 0.0160954 | 0.125298 | 0.199434 | 0.199364 | 0.704359 | 3.89 | 2.52408 | 0.748406 | 16 |
| 39.68547821044922 | Galerkin only | 0.385127 | 49.4623 | - | 0.0225734 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.243152 | 49.4623 | 0.0889549 | 0.0136058 | 0.0136456 | 0.0890226 | 0.154894 | 0.154891 | 0.451216 | 3.89 | 2.52405 | 0.748484 | 16 |
| 45.142704010009766 | Galerkin only | 0.356736 | 45.695 | - | 0.0209462 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.214089 | 45.695 | 0.112096 | 0.0120256 | 0.012062 | 0.11207 | 0.121229 | 0.121238 | 0.333008 | 3.92 | 2.57107 | 0.743336 | 16 |
| 47.081356048583984 | Galerkin only | 2.65753 | 67.4831 | - | 0.329896 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 0.719227 | 67.4831 | 0.111566 | 0.0624213 | 0.0622626 | 0.111953 | 0.396328 | 0.396413 | 0.155648 | 3.96 | 3.04993 | 0.695127 | 19 |
| 49.02235794067383 | Galerkin only | 2.12087 | 54.6819 | - | 0.25518 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 0.542539 | 54.6819 | 0.12844 | 0.0459633 | 0.0461077 | 0.128762 | 0.171849 | 0.171826 | 0.142865 | 3.94 | 3.02854 | 0.697163 | 18 |
| 51.78644943237305 | Galerkin only | 3.13795 | 588.069 | - | 1.86061 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 0.651148 | 588.069 | 0.344604 | 0.272929 | 0.272887 | 0.343736 | 1.32369 | 1.32382 | 2.20781 | 3.99 | 2.88177 | 0.711334 | 16 |
| 70.31463623046875 | Galerkin only | 0.21882 | 1.42014 | - | 0.33963 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.20899 | 1.42014 | 0.0551318 | 0.1594 | 0.0624887 | 0.074885 | 0.423478 | 0.11966 | 0.16221 | 4.5 | 1.63258 | 1.17048 | 7 |
| 100.35224914550781 | Galerkin only | 0.270851 | 1.01569 | - | 0.410135 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.230016 | 1.01569 | 0.014536 | 0.156433 | 0.0255665 | 0.0303382 | 0.463858 | 0.0429375 | 0.0531848 | 4.63 | 2.46866 | 1.17241 | 13 |
| 149.05923461914062 | Galerkin only | 0.295979 | 0.830144 | - | 0.443647 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.215924 | 0.830144 | 0.0192726 | 0.139118 | 0.0188008 | 0.0283156 | 0.437447 | 0.0280086 | 0.0387218 | 4.68 | 1.73622 | 1.12222 | 9 |
| 189.86227416992188 | Galerkin only | 0.304632 | 0.714768 | - | 0.45917 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.192536 | 0.714768 | 0.0208002 | 0.148019 | 0.031113 | 0.0412622 | 0.480891 | 0.0304143 | 0.0397712 | 4.44 | 1.42337 | 1.0069 | 9 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | True | [0.820, 0.000, 0.180] | [0.820, 0.000, 0.180] | 0 |
| 32.74006652832031 | True | [0.820, 0.000, 0.180] | [0.820, 0.000, 0.180] | 0 |
| 39.68547821044922 | True | [0.820, 0.000, 0.180] | [0.820, 0.000, 0.180] | 0 |
| 45.142704010009766 | True | [0.836, 0.000, 0.164] | [0.836, 0.000, 0.164] | 0 |
| 47.081356048583984 | True | [0.987, 0.000, 0.013] | [0.987, 0.000, 0.013] | 0 |
| 49.02235794067383 | True | [0.981, 0.000, 0.019] | [0.981, 0.000, 0.019] | 0 |
| 51.78644943237305 | True | [0.937, 0.000, 0.063] | [0.937, 0.000, 0.063] | 0 |
| 70.31463623046875 | True | [0.601, 0.399, 0.000] | [0.601, 0.399, 0.000] | 0 |
| 100.35224914550781 | True | [0.019, 0.981, 0.000] | [0.019, 0.981, 0.000] | 0 |
| 149.05923461914062 | True | [0.241, 0.684, 0.076] | [0.241, 0.684, 0.076] | 0 |
| 189.86227416992188 | True | [0.411, 0.380, 0.209] | [0.411, 0.380, 0.209] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.426625 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 32.74006652832031 | 0.42389 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 39.68547821044922 | 0.422726 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 45.142704010009766 | 0.422049 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 47.081356048583984 | 0.446986 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 49.02235794067383 | 0.443766 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 51.78644943237305 | 0.453676 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 70.31463623046875 | 0.441741 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 100.35224914550781 | 0.810729 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 149.05923461914062 | 0.927947 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |
| 189.86227416992188 | 0.883063 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e0 |

Runtime: 68612.45 s.
