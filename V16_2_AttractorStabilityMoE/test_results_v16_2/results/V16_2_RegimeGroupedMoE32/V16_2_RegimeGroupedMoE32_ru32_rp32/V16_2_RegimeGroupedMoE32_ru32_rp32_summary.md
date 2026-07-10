# AttractorMoE-ROM V16 Physics-Generalizable Attractor Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=4, experts_per_group=4, shared_experts_per_group=0, group_top_k=4, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 0 / 1; routed gate floor: 0.

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
| rhs_l2 | 0.329929 | 0.17032 | 0.183024 | 0.676187 |
| pressure_head_l2 | 0.253062 | 0.370559 | 0.0106476 | 1.35506 |
| one_step_a_l2 | 0.0405247 | 0.0522061 | 0.0105682 | 0.199689 |
| one_step_b_l2 | 0.257363 | 0.368846 | 0.0192218 | 1.35748 |
| rollout_a_l2 | 0.284682 | 0.443364 | 0.0238999 | 1.64507 |
| rollout_b_l2 | 0.476438 | 0.482226 | 0.0346918 | 1.57177 |
| one_step_pressure_energy_error | 0.2802 | 0.550998 | 0.00191351 | 1.97877 |
| rollout_pressure_energy_error | 0.610479 | 0.886623 | 0.00171716 | 3.12886 |

Error curve CSV: `/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2/results/V16_2_RegimeGroupedMoE32/V16_2_RegimeGroupedMoE32_ru32_rp32/V16_2_RegimeGroupedMoE32_ru32_rp32_error_vs_re.csv`

Error curve SVG: `/root/moe/V16_2_AttractorStabilityMoE/test_results_v16_2/results/V16_2_RegimeGroupedMoE32/V16_2_RegimeGroupedMoE32_ru32_rp32/V16_2_RegimeGroupedMoE32_ru32_rp32_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24.630435943603516 | Galerkin only | 0.497642 | 47.4738 | - | 0.0292339 | - | - | - | - | - | - | - | - | - |
| 24.630435943603516 | HPRS-MoE | 0.37637 | 47.4738 | 0.454151 | 0.0207941 | 0.0208006 | 0.454798 | 0.267951 | 0.267782 | 1.06148 | 13 | 1.53361 | 1.73897 | 9 |
| 32.74006652832031 | Galerkin only | 0.422785 | 56.3403 | - | 0.0247627 | - | - | - | - | - | - | - | - | - |
| 32.74006652832031 | HPRS-MoE | 0.239739 | 56.3403 | 0.272717 | 0.0134765 | 0.0135019 | 0.273327 | 0.182844 | 0.182819 | 0.950508 | 13 | 1.54735 | 1.72442 | 9 |
| 39.68547821044922 | Galerkin only | 0.385127 | 49.4623 | - | 0.0225734 | - | - | - | - | - | - | - | - | - |
| 39.68547821044922 | HPRS-MoE | 0.204372 | 49.4623 | 0.152279 | 0.0116801 | 0.011699 | 0.152616 | 0.141692 | 0.141731 | 0.512849 | 13 | 1.55498 | 1.71621 | 9 |
| 45.142704010009766 | Galerkin only | 0.356736 | 45.695 | - | 0.0209462 | - | - | - | - | - | - | - | - | - |
| 45.142704010009766 | HPRS-MoE | 0.183024 | 45.695 | 0.143479 | 0.0105561 | 0.0105682 | 0.143629 | 0.108557 | 0.108615 | 0.327804 | 13 | 1.55923 | 1.71162 | 8 |
| 47.081356048583984 | Galerkin only | 2.65753 | 67.4831 | - | 0.329896 | - | - | - | - | - | - | - | - | - |
| 47.081356048583984 | HPRS-MoE | 0.676187 | 67.4831 | 0.169657 | 0.0547892 | 0.0548275 | 0.170023 | 0.393399 | 0.392494 | 0.273057 | 13 | 1.56828 | 1.70084 | 9 |
| 49.02235794067383 | Galerkin only | 2.12087 | 54.6819 | - | 0.25518 | - | - | - | - | - | - | - | - | - |
| 49.02235794067383 | HPRS-MoE | 0.563416 | 54.6819 | 0.152755 | 0.0437547 | 0.0437815 | 0.153004 | 0.214109 | 0.214681 | 0.302306 | 13 | 1.56917 | 1.69991 | 9 |
| 51.78644943237305 | Galerkin only | 3.13795 | 588.069 | - | 1.86061 | - | - | - | - | - | - | - | - | - |
| 51.78644943237305 | HPRS-MoE | 0.533975 | 588.069 | 1.35506 | 0.199714 | 0.199689 | 1.35748 | 1.72374 | 1.64507 | 1.57177 | 13 | 1.56728 | 1.70274 | 8 |
| 70.31463623046875 | Galerkin only | 0.21882 | 1.42014 | - | 0.33963 | - | - | - | - | - | - | - | - | - |
| 70.31463623046875 | HPRS-MoE | 0.206882 | 1.42014 | 0.0385353 | 0.148236 | 0.0379492 | 0.0516363 | 0.401168 | 0.0866085 | 0.108154 | 12 | 0.826929 | 2.13301 | 1 |
| 100.35224914550781 | Galerkin only | 0.270851 | 1.01569 | - | 0.410135 | - | - | - | - | - | - | - | - | - |
| 100.35224914550781 | HPRS-MoE | 0.232009 | 1.01569 | 0.0164292 | 0.16056 | 0.0195174 | 0.0256578 | 0.400704 | 0.0344012 | 0.0549921 | 11.7 | 1.26686 | 1.91475 | 4 |
| 149.05923461914062 | Galerkin only | 0.295979 | 0.830144 | - | 0.443647 | - | - | - | - | - | - | - | - | - |
| 149.05923461914062 | HPRS-MoE | 0.218429 | 0.830144 | 0.0106476 | 0.128963 | 0.0154959 | 0.0192218 | 0.406772 | 0.0238999 | 0.0346918 | 12 | 1.50277 | 1.72319 | 6 |
| 189.86227416992188 | Galerkin only | 0.304632 | 0.714768 | - | 0.45917 | - | - | - | - | - | - | - | - | - |
| 189.86227416992188 | HPRS-MoE | 0.194812 | 0.714768 | 0.0179695 | 0.141672 | 0.0179422 | 0.0295997 | 0.536314 | 0.0334035 | 0.0431995 | 12.3 | 2.01261 | 1.36308 | 8 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 24.630435943603516 | False | - | - | nan |
| 32.74006652832031 | False | - | - | nan |
| 39.68547821044922 | False | - | - | nan |
| 45.142704010009766 | False | - | - | nan |
| 47.081356048583984 | False | - | - | nan |
| 49.02235794067383 | False | - | - | nan |
| 51.78644943237305 | False | - | - | nan |
| 70.31463623046875 | False | - | - | nan |
| 100.35224914550781 | False | - | - | nan |
| 149.05923461914062 | False | - | - | nan |
| 189.86227416992188 | False | - | - | nan |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 24.630435943603516 | 0.898122 | False | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 32.74006652832031 | 0.895503 | False | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 39.68547821044922 | 0.895296 | False | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 45.142704010009766 | 0.89513 | False | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 47.081356048583984 | 0.966236 | True | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 49.02235794067383 | 0.946543 | False | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 51.78644943237305 | 0.997694 | True | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 70.31463623046875 | 0.667066 | False | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 100.35224914550781 | 0.958863 | True | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 149.05923461914062 | 0.959357 | True | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |
| 189.86227416992188 | 0.9427 | False | low_Re_lt_80: e0; mid_Re_80_160: e5; high_Re_ge_160: e5 |

Runtime: 67229.57 s.
