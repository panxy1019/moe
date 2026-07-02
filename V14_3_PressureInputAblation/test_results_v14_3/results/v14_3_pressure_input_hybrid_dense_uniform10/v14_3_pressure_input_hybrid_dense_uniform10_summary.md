# HPRS-MoE-ROM V14_3 Pressure Input Ablation Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Pressure input ablation mode: `hybrid`. `pressure_only` is the unchanged V14 baseline, `velocity_only` feeds `[a_next,0]`, and `hybrid` feeds `[a_next,b_base]` to the unchanged pressure experts.

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
| rhs_l2 | 0.0937745 | 0.0204994 | 0.0698518 | 0.124615 |
| pressure_head_l2 | 0.118311 | 0.290707 | 0.00901378 | 0.988577 |
| one_step_a_l2 | 0.0342301 | 0.0159248 | 0.0198569 | 0.0772327 |
| one_step_b_l2 | 0.133363 | 0.291404 | 0.0206524 | 1.00627 |
| rollout_a_l2 | 0.18517 | 0.0659017 | 0.113702 | 0.350866 |
| rollout_b_l2 | 0.29243 | 0.29775 | 0.135839 | 1.17979 |
| one_step_pressure_energy_error | 0.059801 | 0.123896 | 0.00163462 | 0.428328 |
| rollout_pressure_energy_error | 0.150939 | 0.115266 | 0.0148936 | 0.442185 |

Error curve CSV: `/root/moe/V14_3_PressureInputAblation/test_results_v14_3/results/v14_3_pressure_input_hybrid_dense_uniform10/v14_3_pressure_input_hybrid_dense_uniform10_error_vs_re.csv`

Error curve SVG: `/root/moe/V14_3_PressureInputAblation/test_results_v14_3/results/v14_3_pressure_input_hybrid_dense_uniform10/v14_3_pressure_input_hybrid_dense_uniform10_error_vs_re.svg`

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50.0 | Galerkin only | 0.21517 | 1.32573 | - | 0.338327 | - | - | - | - | - | - | - | - | - |
| 50.0 | HPRS-MoE | 0.121243 | 1.32573 | 0.988577 | 0.161048 | 0.0772327 | 1.00627 | 1.57308 | 0.350866 | 1.17979 | 3.06 | 2.75366 | 0.983804 | 18 |
| 78.09062957763672 | Galerkin only | 0.196613 | 0.870268 | - | 0.280223 | - | - | - | - | - | - | - | - | - |
| 78.09062957763672 | HPRS-MoE | 0.11886 | 0.870268 | 0.0326446 | 0.10915 | 0.0410308 | 0.0566661 | 0.692381 | 0.202631 | 0.218408 | 3.43 | 2.2897 | 1.01261 | 15 |
| 105.98314666748047 | Galerkin only | 0.182201 | 0.561152 | - | 0.292866 | - | - | - | - | - | - | - | - | - |
| 105.98314666748047 | HPRS-MoE | 0.124615 | 0.561152 | 0.0185123 | 0.125251 | 0.0313564 | 0.0374667 | 0.864466 | 0.16344 | 0.186274 | 4.1 | 2.56649 | 1.15153 | 16 |
| 132.74302673339844 | Galerkin only | 0.167073 | 0.427656 | - | 0.29154 | - | - | - | - | - | - | - | - | - |
| 132.74302673339844 | HPRS-MoE | 0.104555 | 0.427656 | 0.0135603 | 0.116876 | 0.0289475 | 0.0302151 | 0.858813 | 0.168677 | 0.193097 | 3.9 | 2.51263 | 1.11985 | 15 |
| 160.78543090820312 | Galerkin only | 0.157965 | 0.342647 | - | 0.299659 | - | - | - | - | - | - | - | - | - |
| 160.78543090820312 | HPRS-MoE | 0.0899786 | 0.342647 | 0.0152942 | 0.110459 | 0.0311093 | 0.035262 | 0.854437 | 0.214001 | 0.243618 | 4.04 | 1.66655 | 1.05301 | 10 |
| 187.2852325439453 | Galerkin only | 0.153562 | 0.309063 | - | 0.306075 | - | - | - | - | - | - | - | - | - |
| 187.2852325439453 | HPRS-MoE | 0.0826106 | 0.309063 | 0.0106245 | 0.109856 | 0.0198569 | 0.0228547 | 0.812935 | 0.185243 | 0.240316 | 3.92 | 2.5204 | 1.03725 | 15 |
| 215.25559997558594 | Galerkin only | 0.153056 | 0.326437 | - | 0.313842 | - | - | - | - | - | - | - | - | - |
| 215.25559997558594 | HPRS-MoE | 0.0787979 | 0.326437 | 0.00901378 | 0.110618 | 0.0199296 | 0.0206524 | 0.899238 | 0.123758 | 0.160587 | 3.74 | 2.52687 | 1.00457 | 15 |
| 244.3544158935547 | Galerkin only | 0.150895 | 0.302368 | - | 0.31893 | - | - | - | - | - | - | - | - | - |
| 244.3544158935547 | HPRS-MoE | 0.0747678 | 0.302368 | 0.0107809 | 0.111326 | 0.0236426 | 0.0212571 | 0.996591 | 0.117912 | 0.154362 | 3.82 | 2.5226 | 1.03179 | 15 |
| 274.376708984375 | Galerkin only | 0.159125 | 0.323393 | - | 0.324344 | - | - | - | - | - | - | - | - | - |
| 274.376708984375 | HPRS-MoE | 0.0724638 | 0.323393 | 0.00945457 | 0.113295 | 0.0286329 | 0.029654 | 1.04238 | 0.113702 | 0.135839 | 3.98 | 2.51533 | 1.06587 | 15 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - | - |
| 300.0 | HPRS-MoE | 0.0698518 | 0.337899 | 0.0746504 | 0.114812 | 0.0405624 | 0.0733236 | 1.03419 | 0.211474 | 0.21201 | 3.95 | 2.5273 | 1.08211 | 16 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 50.0 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 78.09062957763672 | True | [0.841, 0.159, 0.000] | [0.841, 0.159, 0.000] | 0 |
| 105.98314666748047 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 132.74302673339844 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 160.78543090820312 | True | [0.000, 0.488, 0.512] | [0.000, 0.488, 0.512] | 0 |
| 187.2852325439453 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 215.25559997558594 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 244.3544158935547 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 274.376708984375 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |
| 300.0 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 50.0 | 0.874615 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 78.09062957763672 | 0.959803 | True | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 105.98314666748047 | 0.898912 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 132.74302673339844 | 0.72012 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 160.78543090820312 | 0.442108 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 187.2852325439453 | 0.598942 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 215.25559997558594 | 0.595303 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 244.3544158935547 | 0.725807 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 274.376708984375 | 0.765607 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 300.0 | 0.80616 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 22796.14 s.
