# HPRS-MoE-ROM V14_2 Data Ablation Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Data Ablation Split

Test selection: `uniform`, time stride=5, Re stride=2.

- Train Re count: 45
- Test Re count: 10
- Excluded Re count from Re sparsity: 45
- Dense train samples before time sparsity: 4983
- Kept train samples: 1023
- Validation samples: 675
- Test samples: 1255
- Compression vs dense train: 0.205298
- Compression vs all non-test candidates: 0.0904189

| Re | role | total | dense train | kept train | val | test |
|---:|---|---:|---:|---:|---:|---:|
| 50 | test | 126 | 0 | 0 | 0 | 126 |
| 50.2847 | train | 126 | 111 | 23 | 15 | 0 |
| 50.7258 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 51.2548 | train | 126 | 111 | 23 | 15 | 0 |
| 51.8502 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 52.5007 | train | 126 | 111 | 23 | 15 | 0 |
| 53.1986 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 53.9385 | train | 126 | 111 | 23 | 15 | 0 |
| 54.7165 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 55.5294 | train | 125 | 110 | 22 | 15 | 0 |
| 56.3745 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 57.2498 | train | 126 | 111 | 23 | 15 | 0 |
| 58.1535 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 59.0839 | train | 126 | 111 | 23 | 15 | 0 |
| 60.0397 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 61.0197 | train | 126 | 111 | 23 | 15 | 0 |
| 62.0229 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 63.0483 | train | 126 | 111 | 23 | 15 | 0 |
| 64.095 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 65.1623 | train | 125 | 110 | 22 | 15 | 0 |
| 66.2494 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 67.3558 | train | 125 | 110 | 22 | 15 | 0 |
| 68.4807 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 69.6237 | train | 125 | 110 | 22 | 15 | 0 |
| 70.7842 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 71.9617 | train | 126 | 111 | 23 | 15 | 0 |
| 73.1559 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 74.3663 | train | 125 | 110 | 22 | 15 | 0 |
| 75.5924 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 76.834 | train | 126 | 111 | 23 | 15 | 0 |
| 78.0906 | test | 126 | 0 | 0 | 0 | 126 |
| 79.3621 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 80.6479 | train | 126 | 111 | 23 | 15 | 0 |
| 81.9479 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 83.2617 | train | 126 | 111 | 23 | 15 | 0 |
| 84.5891 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 85.9299 | train | 126 | 111 | 23 | 15 | 0 |
| 87.2838 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 88.6505 | train | 125 | 110 | 22 | 15 | 0 |
| 90.03 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 91.4218 | train | 125 | 110 | 22 | 15 | 0 |
| 92.8259 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 94.242 | train | 126 | 111 | 23 | 15 | 0 |
| 95.6699 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 97.1095 | train | 126 | 111 | 23 | 15 | 0 |
| 98.5607 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 100.023 | train | 126 | 111 | 23 | 15 | 0 |
| 101.497 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 102.981 | train | 126 | 111 | 23 | 15 | 0 |
| 104.477 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 105.983 | test | 125 | 0 | 0 | 0 | 125 |
| 107.5 | train | 126 | 111 | 23 | 15 | 0 |
| 109.027 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 110.565 | train | 125 | 110 | 22 | 15 | 0 |
| 112.113 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 113.67 | train | 126 | 111 | 23 | 15 | 0 |
| 115.238 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 116.816 | train | 126 | 111 | 23 | 15 | 0 |
| 118.403 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 120 | train | 126 | 111 | 23 | 15 | 0 |
| 122.588 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 125.742 | train | 126 | 111 | 23 | 15 | 0 |
| 129.154 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 132.743 | test | 125 | 0 | 0 | 0 | 125 |
| 136.471 | train | 126 | 111 | 23 | 15 | 0 |
| 140.313 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 144.253 | train | 125 | 110 | 22 | 15 | 0 |
| 148.279 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 152.38 | train | 126 | 111 | 23 | 15 | 0 |
| 156.551 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 160.785 | test | 125 | 0 | 0 | 0 | 125 |
| 165.078 | train | 126 | 111 | 23 | 15 | 0 |
| 169.424 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 173.821 | train | 125 | 110 | 22 | 15 | 0 |
| 178.265 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 182.754 | train | 126 | 111 | 23 | 15 | 0 |
| 187.285 | test | 126 | 0 | 0 | 0 | 126 |
| 191.857 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 196.466 | train | 126 | 111 | 23 | 15 | 0 |
| 201.113 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 205.794 | train | 126 | 111 | 23 | 15 | 0 |
| 210.509 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 215.256 | test | 126 | 0 | 0 | 0 | 126 |
| 220.034 | train | 125 | 110 | 22 | 15 | 0 |
| 224.842 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 229.679 | train | 125 | 110 | 22 | 15 | 0 |
| 234.544 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 239.436 | train | 126 | 111 | 23 | 15 | 0 |
| 244.354 | test | 125 | 0 | 0 | 0 | 125 |
| 249.298 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 254.267 | train | 126 | 111 | 23 | 15 | 0 |
| 259.26 | excluded_by_re_sparsity | 125 | 0 | 0 | 0 | 0 |
| 264.276 | train | 126 | 111 | 23 | 15 | 0 |
| 269.315 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 274.377 | test | 125 | 0 | 0 | 0 | 125 |
| 279.46 | train | 126 | 111 | 23 | 15 | 0 |
| 284.564 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 289.689 | train | 126 | 111 | 23 | 15 | 0 |
| 294.835 | excluded_by_re_sparsity | 126 | 0 | 0 | 0 | 0 |
| 300 | test | 126 | 0 | 0 | 0 | 126 |

## Aggregate Held-out Metrics

| Metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| rhs_l2 | 0.10193 | 0.0148351 | 0.0872202 | 0.133825 |
| pressure_head_l2 | 0.165872 | 0.248941 | 0.0397648 | 0.90711 |
| one_step_a_l2 | 0.0622635 | 0.007831 | 0.0462246 | 0.077505 |
| one_step_b_l2 | 0.194114 | 0.241648 | 0.0940363 | 0.916376 |
| rollout_a_l2 | 0.237661 | 0.0437422 | 0.14879 | 0.301741 |
| rollout_b_l2 | 0.3519 | 0.203137 | 0.196719 | 0.93236 |

Error curve CSV: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_2_test2_time_s5_re_s2_uniform10/v14_2_test2_time_s5_re_s2_uniform10_error_vs_re.csv`

Error curve SVG: `/root/moe/V14_2_DataAblation/test_results_v14_2/results/v14_2_test2_time_s5_re_s2_uniform10/v14_2_test2_time_s5_re_s2_uniform10_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.0 | Galerkin only | 0.21517 | 1.32573 | - | 0.338327 | - | - | - | - | - | - | - | - | - |
| 50.0 | HPRS-MoE | 0.133825 | 1.32573 | 0.90711 | 0.0712919 | 0.077505 | 0.916376 | 0.232827 | 0.301741 | 0.93236 | 3.71 | 3.00918 | 0.797895 | 18 |
| 78.09062957763672 | Galerkin only | 0.196613 | 0.870268 | - | 0.280223 | - | - | - | - | - | - | - | - | - |
| 78.09062957763672 | HPRS-MoE | 0.117483 | 0.870268 | 0.0805054 | 0.0911653 | 0.0462246 | 0.105032 | 0.300773 | 0.14879 | 0.196719 | 3.98 | 1.86237 | 0.972555 | 14 |
| 105.98314666748047 | Galerkin only | 0.182201 | 0.561152 | - | 0.292866 | - | - | - | - | - | - | - | - | - |
| 105.98314666748047 | HPRS-MoE | 0.115504 | 0.561152 | 0.0397648 | 0.134971 | 0.056987 | 0.100114 | 0.656653 | 0.192877 | 0.197129 | 4.34 | 2.52855 | 1.17685 | 15 |
| 132.74302673339844 | Galerkin only | 0.167073 | 0.427656 | - | 0.29154 | - | - | - | - | - | - | - | - | - |
| 132.74302673339844 | HPRS-MoE | 0.104644 | 0.427656 | 0.0475378 | 0.122736 | 0.0605552 | 0.101999 | 0.772624 | 0.271954 | 0.314729 | 4.32 | 2.25527 | 1.16812 | 11 |
| 160.78543090820312 | Galerkin only | 0.157965 | 0.342647 | - | 0.299659 | - | - | - | - | - | - | - | - | - |
| 160.78543090820312 | HPRS-MoE | 0.0922797 | 0.342647 | 0.066519 | 0.112535 | 0.0583339 | 0.10011 | 0.748425 | 0.269084 | 0.361094 | 4.41 | 1.63772 | 1.15926 | 8 |
| 187.2852325439453 | Galerkin only | 0.153562 | 0.309063 | - | 0.306075 | - | - | - | - | - | - | - | - | - |
| 187.2852325439453 | HPRS-MoE | 0.0883965 | 0.309063 | 0.0715129 | 0.11335 | 0.0594538 | 0.0940363 | 0.689181 | 0.244003 | 0.277773 | 4.4 | 2.13669 | 1.12148 | 11 |
| 215.25559997558594 | Galerkin only | 0.153056 | 0.326437 | - | 0.313842 | - | - | - | - | - | - | - | - | - |
| 215.25559997558594 | HPRS-MoE | 0.0885746 | 0.326437 | 0.0964964 | 0.114207 | 0.063174 | 0.111782 | 0.727608 | 0.200465 | 0.253188 | 4.52 | 2.17303 | 1.1116 | 11 |
| 244.3544158935547 | Galerkin only | 0.150895 | 0.302368 | - | 0.31893 | - | - | - | - | - | - | - | - | - |
| 244.3544158935547 | HPRS-MoE | 0.094979 | 0.302368 | 0.0798194 | 0.112007 | 0.0646617 | 0.110298 | 0.77064 | 0.223782 | 0.276348 | 4.53 | 2.12564 | 1.09559 | 10 |
| 274.376708984375 | Galerkin only | 0.159125 | 0.323393 | - | 0.324344 | - | - | - | - | - | - | - | - | - |
| 274.376708984375 | HPRS-MoE | 0.0963942 | 0.323393 | 0.125898 | 0.104678 | 0.0675869 | 0.133499 | 0.755465 | 0.252908 | 0.303601 | 4.52 | 2.17375 | 1.09669 | 11 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - | - |
| 300.0 | HPRS-MoE | 0.0872202 | 0.337899 | 0.143552 | 0.100099 | 0.0681524 | 0.167897 | 0.723455 | 0.271003 | 0.40606 | 4.58 | 2.25831 | 1.0931 | 12 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 50.0 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 78.09062957763672 | True | [0.508, 0.492, 0.000] | [0.508, 0.492, 0.000] | 0 |
| 105.98314666748047 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 132.74302673339844 | True | [0.000, 0.928, 0.072] | [0.000, 0.928, 0.072] | 0 |
| 160.78543090820312 | True | [0.024, 0.344, 0.632] | [0.024, 0.344, 0.632] | 0 |
| 187.2852325439453 | True | [0.016, 0.103, 0.881] | [0.016, 0.103, 0.881] | 0 |
| 215.25559997558594 | True | [0.032, 0.071, 0.897] | [0.032, 0.071, 0.897] | 0 |
| 244.3544158935547 | True | [0.032, 0.088, 0.880] | [0.032, 0.088, 0.880] | 0 |
| 274.376708984375 | True | [0.000, 0.104, 0.896] | [0.000, 0.104, 0.896] | 0 |
| 300.0 | True | [0.008, 0.063, 0.929] | [0.008, 0.063, 0.929] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 50.0 | 0.745785 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 78.09062957763672 | 0.672838 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 105.98314666748047 | 0.564447 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 132.74302673339844 | 0.648194 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 160.78543090820312 | 0.666912 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 187.2852325439453 | 0.577305 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 215.25559997558594 | 0.620004 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 244.3544158935547 | 0.657282 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 274.376708984375 | 0.631037 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 300.0 | 0.636456 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 3802.46 s.
