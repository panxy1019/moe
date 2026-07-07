# HPRS-MoE-ROM V14_3 Pressure Input Ablation Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Pressure input ablation mode: `pressure_only`. `pressure_only` is the unchanged V14 baseline, `velocity_only` feeds `[a_next,0]`, and `hybrid` feeds `[a_next,b_base]` to the unchanged pressure experts.

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
| rhs_l2 | 0.0925724 | 0.0182522 | 0.0707683 | 0.120913 |
| pressure_head_l2 | 0.107682 | 0.283605 | 0.00419661 | 0.957718 |
| one_step_a_l2 | 0.0230022 | 0.0161161 | 0.00925768 | 0.0662447 |
| one_step_b_l2 | 0.121132 | 0.283365 | 0.015303 | 0.970201 |
| rollout_a_l2 | 0.073183 | 0.0786007 | 0.0267882 | 0.302943 |
| rollout_b_l2 | 0.164159 | 0.332395 | 0.0298593 | 1.1591 |
| one_step_pressure_energy_error | 0.0445221 | 0.108348 | 0.00343147 | 0.368951 |
| rollout_pressure_energy_error | 0.0721308 | 0.123567 | 0.000551049 | 0.432554 |

Error curve CSV: `/root/moe/V14_3_PressureInputAblation/test_results_v14_3/results/v14_3_pressure_input_pressure_only_dense_uniform10/v14_3_pressure_input_pressure_only_dense_uniform10_error_vs_re.csv`

Error curve SVG: `/root/moe/V14_3_PressureInputAblation/test_results_v14_3/results/v14_3_pressure_input_pressure_only_dense_uniform10/v14_3_pressure_input_pressure_only_dense_uniform10_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.0 | Galerkin only | 0.21517 | 1.32573 | - | 0.338327 | - | - | - | - | - | - | - | - | - |
| 50.0 | HPRS-MoE | 0.110896 | 1.32573 | 0.957718 | 0.158978 | 0.0662447 | 0.970201 | 1.50082 | 0.302943 | 1.1591 | 5 | 2.6719 | 1.08935 | 17 |
| 78.09062957763672 | Galerkin only | 0.196613 | 0.870268 | - | 0.280223 | - | - | - | - | - | - | - | - | - |
| 78.09062957763672 | HPRS-MoE | 0.119212 | 0.870268 | 0.023806 | 0.10375 | 0.0331059 | 0.0561346 | 0.725952 | 0.0752552 | 0.0796697 | 5 | 2.40558 | 1.14107 | 14 |
| 105.98314666748047 | Galerkin only | 0.182201 | 0.561152 | - | 0.292866 | - | - | - | - | - | - | - | - | - |
| 105.98314666748047 | HPRS-MoE | 0.120913 | 0.561152 | 0.00732601 | 0.102046 | 0.0108983 | 0.0171477 | 0.763986 | 0.0267882 | 0.0298593 | 4.98 | 2.48058 | 1.27434 | 14 |
| 132.74302673339844 | Galerkin only | 0.167073 | 0.427656 | - | 0.29154 | - | - | - | - | - | - | - | - | - |
| 132.74302673339844 | HPRS-MoE | 0.101772 | 0.427656 | 0.00976709 | 0.0971215 | 0.00925768 | 0.015303 | 0.750621 | 0.040255 | 0.0481885 | 4.86 | 2.44076 | 1.27666 | 14 |
| 160.78543090820312 | Galerkin only | 0.157965 | 0.342647 | - | 0.299659 | - | - | - | - | - | - | - | - | - |
| 160.78543090820312 | HPRS-MoE | 0.0901878 | 0.342647 | 0.0117305 | 0.0988679 | 0.0142408 | 0.0200866 | 0.814068 | 0.0453741 | 0.0546321 | 4.08 | 2.02183 | 1.12517 | 11 |
| 187.2852325439453 | Galerkin only | 0.153562 | 0.309063 | - | 0.306075 | - | - | - | - | - | - | - | - | - |
| 187.2852325439453 | HPRS-MoE | 0.083739 | 0.309063 | 0.00555226 | 0.101544 | 0.0150408 | 0.0178647 | 0.855446 | 0.0484827 | 0.0537741 | 4.14 | 2.53004 | 1.09623 | 15 |
| 215.25559997558594 | Galerkin only | 0.153056 | 0.326437 | - | 0.313842 | - | - | - | - | - | - | - | - | - |
| 215.25559997558594 | HPRS-MoE | 0.0796444 | 0.326437 | 0.00571255 | 0.103729 | 0.0140574 | 0.0183753 | 0.87951 | 0.0352075 | 0.0353857 | 4.07 | 2.54822 | 1.06386 | 14 |
| 244.3544158935547 | Galerkin only | 0.150895 | 0.302368 | - | 0.31893 | - | - | - | - | - | - | - | - | - |
| 244.3544158935547 | HPRS-MoE | 0.0756606 | 0.302368 | 0.00419661 | 0.107899 | 0.0174609 | 0.0206708 | 0.945222 | 0.0326137 | 0.0318397 | 4.01 | 2.49501 | 1.10289 | 14 |
| 274.376708984375 | Galerkin only | 0.159125 | 0.323393 | - | 0.324344 | - | - | - | - | - | - | - | - | - |
| 274.376708984375 | HPRS-MoE | 0.0729293 | 0.323393 | 0.0053628 | 0.11064 | 0.0208506 | 0.0240637 | 0.970049 | 0.0395604 | 0.0429471 | 4.08 | 2.4901 | 1.11804 | 14 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - | - |
| 300.0 | HPRS-MoE | 0.0707683 | 0.337899 | 0.0456453 | 0.112188 | 0.0288649 | 0.051472 | 0.956126 | 0.0853497 | 0.106186 | 4.08 | 2.50915 | 1.10596 | 14 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 50.0 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 78.09062957763672 | True | [0.921, 0.079, 0.000] | [0.921, 0.079, 0.000] | 0 |
| 105.98314666748047 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 132.74302673339844 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 160.78543090820312 | True | [0.000, 0.200, 0.800] | [0.000, 0.200, 0.800] | 0 |
| 187.2852325439453 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 215.25559997558594 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 244.3544158935547 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 274.376708984375 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 300.0 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 50.0 | 0.730702 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 78.09062957763672 | 0.898056 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 105.98314666748047 | 0.678279 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 132.74302673339844 | 0.623893 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 160.78543090820312 | 0.58798 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 187.2852325439453 | 0.429501 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 215.25559997558594 | 0.468962 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 244.3544158935547 | 0.485617 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 274.376708984375 | 0.528444 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 300.0 | 0.621112 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 25568.97 s.
