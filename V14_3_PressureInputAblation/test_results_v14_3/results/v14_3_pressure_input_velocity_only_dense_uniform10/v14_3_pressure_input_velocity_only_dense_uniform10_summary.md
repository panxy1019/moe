# HPRS-MoE-ROM V14_3 Pressure Input Ablation Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Pressure input ablation mode: `velocity_only`. `pressure_only` is the unchanged V14 baseline, `velocity_only` feeds `[a_next,0]`, and `hybrid` feeds `[a_next,b_base]` to the unchanged pressure experts.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Dense Training Split

Test selection: `uniform`, time stride=1, Re stride=1.

- Train Re count: 90
- Test Re count: 10
- Excluded Re count from Re sparsity: 0
- Dense train samples before time sparsity: 9964
- Kept train samples: 9964
- Validation samples: 1350
- Test samples: 1255
- Compression vs dense train: 1
- Compression vs all non-test candidates: 0.880679

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 50 | test | 126 | 0 | 0 | 0 | 126 |
| 50.2847 | train | 126 | 111 | 111 | 15 | 0 |
| 50.7258 | train | 125 | 110 | 110 | 15 | 0 |
| 51.2548 | train | 126 | 111 | 111 | 15 | 0 |
| 51.8502 | train | 126 | 111 | 111 | 15 | 0 |
| 52.5007 | train | 126 | 111 | 111 | 15 | 0 |
| 53.1986 | train | 125 | 110 | 110 | 15 | 0 |
| 53.9385 | train | 126 | 111 | 111 | 15 | 0 |
| 54.7165 | train | 126 | 111 | 111 | 15 | 0 |
| 55.5294 | train | 125 | 110 | 110 | 15 | 0 |
| 56.3745 | train | 126 | 111 | 111 | 15 | 0 |
| 57.2498 | train | 126 | 111 | 111 | 15 | 0 |
| 58.1535 | train | 125 | 110 | 110 | 15 | 0 |
| 59.0839 | train | 126 | 111 | 111 | 15 | 0 |
| 60.0397 | train | 126 | 111 | 111 | 15 | 0 |
| 61.0197 | train | 126 | 111 | 111 | 15 | 0 |
| 62.0229 | train | 125 | 110 | 110 | 15 | 0 |
| 63.0483 | train | 126 | 111 | 111 | 15 | 0 |
| 64.095 | train | 126 | 111 | 111 | 15 | 0 |
| 65.1623 | train | 125 | 110 | 110 | 15 | 0 |
| 66.2494 | train | 125 | 110 | 110 | 15 | 0 |
| 67.3558 | train | 125 | 110 | 110 | 15 | 0 |
| 68.4807 | train | 126 | 111 | 111 | 15 | 0 |
| 69.6237 | train | 125 | 110 | 110 | 15 | 0 |
| 70.7842 | train | 125 | 110 | 110 | 15 | 0 |
| 71.9617 | train | 126 | 111 | 111 | 15 | 0 |
| 73.1559 | train | 126 | 111 | 111 | 15 | 0 |
| 74.3663 | train | 125 | 110 | 110 | 15 | 0 |
| 75.5924 | train | 126 | 111 | 111 | 15 | 0 |
| 76.834 | train | 126 | 111 | 111 | 15 | 0 |
| 78.0906 | test | 126 | 0 | 0 | 0 | 126 |
| 79.3621 | train | 126 | 111 | 111 | 15 | 0 |
| 80.6479 | train | 126 | 111 | 111 | 15 | 0 |
| 81.9479 | train | 126 | 111 | 111 | 15 | 0 |
| 83.2617 | train | 126 | 111 | 111 | 15 | 0 |
| 84.5891 | train | 126 | 111 | 111 | 15 | 0 |
| 85.9299 | train | 126 | 111 | 111 | 15 | 0 |
| 87.2838 | train | 126 | 111 | 111 | 15 | 0 |
| 88.6505 | train | 125 | 110 | 110 | 15 | 0 |
| 90.03 | train | 126 | 111 | 111 | 15 | 0 |
| 91.4218 | train | 125 | 110 | 110 | 15 | 0 |
| 92.8259 | train | 126 | 111 | 111 | 15 | 0 |
| 94.242 | train | 126 | 111 | 111 | 15 | 0 |
| 95.6699 | train | 125 | 110 | 110 | 15 | 0 |
| 97.1095 | train | 126 | 111 | 111 | 15 | 0 |
| 98.5607 | train | 125 | 110 | 110 | 15 | 0 |
| 100.023 | train | 126 | 111 | 111 | 15 | 0 |
| 101.497 | train | 126 | 111 | 111 | 15 | 0 |
| 102.981 | train | 126 | 111 | 111 | 15 | 0 |
| 104.477 | train | 126 | 111 | 111 | 15 | 0 |
| 105.983 | test | 125 | 0 | 0 | 0 | 125 |
| 107.5 | train | 126 | 111 | 111 | 15 | 0 |
| 109.027 | train | 125 | 110 | 110 | 15 | 0 |
| 110.565 | train | 125 | 110 | 110 | 15 | 0 |
| 112.113 | train | 126 | 111 | 111 | 15 | 0 |
| 113.67 | train | 126 | 111 | 111 | 15 | 0 |
| 115.238 | train | 126 | 111 | 111 | 15 | 0 |
| 116.816 | train | 126 | 111 | 111 | 15 | 0 |
| 118.403 | train | 126 | 111 | 111 | 15 | 0 |
| 120 | train | 126 | 111 | 111 | 15 | 0 |
| 122.588 | train | 126 | 111 | 111 | 15 | 0 |
| 125.742 | train | 126 | 111 | 111 | 15 | 0 |
| 129.154 | train | 126 | 111 | 111 | 15 | 0 |
| 132.743 | test | 125 | 0 | 0 | 0 | 125 |
| 136.471 | train | 126 | 111 | 111 | 15 | 0 |
| 140.313 | train | 125 | 110 | 110 | 15 | 0 |
| 144.253 | train | 125 | 110 | 110 | 15 | 0 |
| 148.279 | train | 126 | 111 | 111 | 15 | 0 |
| 152.38 | train | 126 | 111 | 111 | 15 | 0 |
| 156.551 | train | 126 | 111 | 111 | 15 | 0 |
| 160.785 | test | 125 | 0 | 0 | 0 | 125 |
| 165.078 | train | 126 | 111 | 111 | 15 | 0 |
| 169.424 | train | 126 | 111 | 111 | 15 | 0 |
| 173.821 | train | 125 | 110 | 110 | 15 | 0 |
| 178.265 | train | 126 | 111 | 111 | 15 | 0 |
| 182.754 | train | 126 | 111 | 111 | 15 | 0 |
| 187.285 | test | 126 | 0 | 0 | 0 | 126 |
| 191.857 | train | 126 | 111 | 111 | 15 | 0 |
| 196.466 | train | 126 | 111 | 111 | 15 | 0 |
| 201.113 | train | 125 | 110 | 110 | 15 | 0 |
| 205.794 | train | 126 | 111 | 111 | 15 | 0 |
| 210.509 | train | 126 | 111 | 111 | 15 | 0 |
| 215.256 | test | 126 | 0 | 0 | 0 | 126 |
| 220.034 | train | 125 | 110 | 110 | 15 | 0 |
| 224.842 | train | 125 | 110 | 110 | 15 | 0 |
| 229.679 | train | 125 | 110 | 110 | 15 | 0 |
| 234.544 | train | 125 | 110 | 110 | 15 | 0 |
| 239.436 | train | 126 | 111 | 111 | 15 | 0 |
| 244.354 | test | 125 | 0 | 0 | 0 | 125 |
| 249.298 | train | 126 | 111 | 111 | 15 | 0 |
| 254.267 | train | 126 | 111 | 111 | 15 | 0 |
| 259.26 | train | 125 | 110 | 110 | 15 | 0 |
| 264.276 | train | 126 | 111 | 111 | 15 | 0 |
| 269.315 | train | 126 | 111 | 111 | 15 | 0 |
| 274.377 | test | 125 | 0 | 0 | 0 | 125 |
| 279.46 | train | 126 | 111 | 111 | 15 | 0 |
| 284.564 | train | 126 | 111 | 111 | 15 | 0 |
| 289.689 | train | 126 | 111 | 111 | 15 | 0 |
| 294.835 | train | 126 | 111 | 111 | 15 | 0 |
| 300 | test | 126 | 0 | 0 | 0 | 126 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 0.0916761 | 0.0178004 | 0.0700769 | 0.120967 |
| pressure_head_l2 | 0.114961 | 0.292553 | 0.00630476 | 0.99086 |
| one_step_a_l2 | 0.0259003 | 0.014864 | 0.0111692 | 0.0656736 |
| one_step_b_l2 | 0.126876 | 0.290914 | 0.0139873 | 0.998312 |
| rollout_a_l2 | 0.0998976 | 0.0730735 | 0.0361559 | 0.28161 |
| rollout_b_l2 | 0.208693 | 0.361565 | 0.0381312 | 1.2857 |
| one_step_pressure_energy_error | 0.0449067 | 0.100472 | 0.00354155 | 0.343918 |
| rollout_pressure_energy_error | 0.0669514 | 0.0788961 | 0.00441701 | 0.265514 |

Error curve CSV: `/root/moe/V14_3_PressureInputAblation/test_results_v14_3/results/v14_3_pressure_input_velocity_only_dense_uniform10/v14_3_pressure_input_velocity_only_dense_uniform10_error_vs_re.csv`

Error curve SVG: `/root/moe/V14_3_PressureInputAblation/test_results_v14_3/results/v14_3_pressure_input_velocity_only_dense_uniform10/v14_3_pressure_input_velocity_only_dense_uniform10_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.0 | Galerkin only | 0.21517 | 1.32573 | - | 0.338327 | - | - | - | - | - | - | - | - | - |
| 50.0 | HPRS-MoE | 0.112228 | 1.32573 | 0.99086 | 0.188217 | 0.0656736 | 0.998312 | 1.87202 | 0.28161 | 1.2857 | 4 | 3.04223 | 0.759331 | 18 |
| 78.09062957763672 | Galerkin only | 0.196613 | 0.870268 | - | 0.280223 | - | - | - | - | - | - | - | - | - |
| 78.09062957763672 | HPRS-MoE | 0.112649 | 0.870268 | 0.0333538 | 0.103067 | 0.0322886 | 0.0541517 | 0.685053 | 0.132454 | 0.156418 | 4.03 | 2.8825 | 0.812462 | 17 |
| 105.98314666748047 | Galerkin only | 0.182201 | 0.561152 | - | 0.292866 | - | - | - | - | - | - | - | - | - |
| 105.98314666748047 | HPRS-MoE | 0.120967 | 0.561152 | 0.00723092 | 0.100537 | 0.0111692 | 0.0139873 | 0.773709 | 0.0361559 | 0.0458286 | 4.62 | 2.55268 | 1.23401 | 16 |
| 132.74302673339844 | Galerkin only | 0.167073 | 0.427656 | - | 0.29154 | - | - | - | - | - | - | - | - | - |
| 132.74302673339844 | HPRS-MoE | 0.10211 | 0.427656 | 0.00962678 | 0.0992604 | 0.0122023 | 0.0151626 | 0.848802 | 0.0401997 | 0.0381312 | 4.67 | 2.50881 | 1.25632 | 16 |
| 160.78543090820312 | Galerkin only | 0.157965 | 0.342647 | - | 0.299659 | - | - | - | - | - | - | - | - | - |
| 160.78543090820312 | HPRS-MoE | 0.0888796 | 0.342647 | 0.0131799 | 0.0989875 | 0.0172394 | 0.0220659 | 0.81319 | 0.0770969 | 0.0887429 | 4.77 | 1.62629 | 1.22928 | 9 |
| 187.2852325439453 | Galerkin only | 0.153562 | 0.309063 | - | 0.306075 | - | - | - | - | - | - | - | - | - |
| 187.2852325439453 | HPRS-MoE | 0.0825132 | 0.309063 | 0.00724391 | 0.110906 | 0.0217079 | 0.0230841 | 0.797987 | 0.07676 | 0.0941559 | 4.73 | 2.43926 | 1.23424 | 14 |
| 215.25559997558594 | Galerkin only | 0.153056 | 0.326437 | - | 0.313842 | - | - | - | - | - | - | - | - | - |
| 215.25559997558594 | HPRS-MoE | 0.0792849 | 0.326437 | 0.00705997 | 0.115713 | 0.0230721 | 0.0257374 | 0.950941 | 0.0748205 | 0.087956 | 4.52 | 2.46358 | 1.16035 | 14 |
| 244.3544158935547 | Galerkin only | 0.150895 | 0.302368 | - | 0.31893 | - | - | - | - | - | - | - | - | - |
| 244.3544158935547 | HPRS-MoE | 0.0755488 | 0.302368 | 0.00630476 | 0.117232 | 0.0211603 | 0.0242179 | 0.965931 | 0.05553 | 0.0649563 | 4.21 | 2.49104 | 1.08744 | 15 |
| 274.376708984375 | Galerkin only | 0.159125 | 0.323393 | - | 0.324344 | - | - | - | - | - | - | - | - | - |
| 274.376708984375 | HPRS-MoE | 0.0725042 | 0.323393 | 0.00687257 | 0.11475 | 0.0219837 | 0.0260774 | 0.966021 | 0.0499768 | 0.0502847 | 4.1 | 2.50915 | 1.01311 | 15 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - | - |
| 300.0 | HPRS-MoE | 0.0700769 | 0.337899 | 0.0678759 | 0.11176 | 0.0325057 | 0.065966 | 0.929802 | 0.174372 | 0.174755 | 4.05 | 2.5008 | 1.00274 | 15 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 50.0 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 78.09062957763672 | True | [0.968, 0.032, 0.000] | [0.968, 0.032, 0.000] | 0 |
| 105.98314666748047 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 132.74302673339844 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 160.78543090820312 | True | [0.000, 0.544, 0.456] | [0.000, 0.544, 0.456] | 0 |
| 187.2852325439453 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 215.25559997558594 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 244.3544158935547 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 274.376708984375 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 300.0 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 50.0 | 0.573134 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 78.09062957763672 | 0.873962 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 105.98314666748047 | 0.733966 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 132.74302673339844 | 0.649015 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 160.78543090820312 | 0.618439 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 187.2852325439453 | 0.531078 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 215.25559997558594 | 0.609268 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 244.3544158935547 | 0.656409 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 274.376708984375 | 0.512859 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 300.0 | 0.727758 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 29129.72 s.
