# HPRS-MoE-ROM v14 Summary

## Architecture

Shared encoder + 3 latent refinement blocks, hidden_dim=224, regime_groups=3, experts_per_group=6, shared_experts_per_group=1, group_top_k=1, in_group_top_k=2, expert_hidden=768, expert_blocks=3, quadratic_rank=4.

Shared/routed scales: 1 / 0.85; routed gate floor: 0.

A shared group router selects a physics regime, then group-local velocity/pressure Top-2 routers mix a group-shared expert with routed physics-aware operator experts. Experts output `residual` velocity operator targets plus a pressure `closure` branch. For `residual`, the learned closure is added to the Galerkin RHS before RK4.

Losses: one-step coefficient, sampled reconstruction, full-RHS operator, closed-loop multi-step rollout, energy consistency, trajectory consistency, pressure closure, relative terms, group/router load-balance, entropy, temporal smoothness, expert diversity, and weak Re-regime supervision.

## Metrics

Integrator: `rk4`.

| Test Re | Model | RHS L2 | pressure base L2 | pressure final L2 | TF one-step L2 | Auto a one-step L2 | Auto b one-step L2 | TF rollout mean | Auto a rollout mean | Auto b rollout mean | active experts | load CV | entropy | dead experts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 56.3745231628418 | Galerkin only | 0.232684 | 2.72647 | - | 0.360909 | - | - | - | - | - | - | - | - | - |
| 56.3745231628418 | HPRS-MoE | 0.123469 | 2.72647 | 0.294127 | 0.116862 | 0.0563213 | 0.297137 | 0.810667 | 0.265512 | 0.495149 | 4 | 2.60947 | 1.16471 | 17 |
| 120.0 | Galerkin only | 0.173703 | 0.481705 | - | 0.290683 | - | - | - | - | - | - | - | - | - |
| 120.0 | HPRS-MoE | 0.111732 | 0.481705 | 0.00847165 | 0.121461 | 0.0184185 | 0.0243795 | 0.885083 | 0.0512296 | 0.0500808 | 3.98 | 2.62636 | 1.13991 | 17 |
| 300.0 | Galerkin only | 0.154183 | 0.337899 | - | 0.327571 | - | - | - | - | - | - | - | - | - |
| 300.0 | HPRS-MoE | 0.0707076 | 0.337899 | 0.0451153 | 0.112371 | 0.034579 | 0.0538873 | 0.965551 | 0.216845 | 0.232747 | 4.21 | 2.55623 | 1.08846 | 16 |

## Expert Diagnostics

| Test Re | shared in selected group | group mean load | group top1 fraction | group entropy |
|---:|---|---|---|---:|
| 56.3745231628418 | True | [1.000, 0.000, 0.000] | [1.000, 0.000, 0.000] | 0 |
| 120.0 | True | [0.000, 1.000, 0.000] | [0.000, 1.000, 0.000] | 0 |
| 300.0 | True | [0.000, 0.000, 1.000] | [0.000, 0.000, 1.000] | 0 |

| Test Re | max |cos(expert_i, expert_j)| | collapse flag | low/mid/high train top experts |
|---:|---:|---|---|
| 56.3745231628418 | 0.867166 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 120.0 | 0.716826 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |
| 300.0 | 0.523755 | False | low_Re_lt_80: e0; mid_Re_80_160: e7; high_Re_ge_160: e14 |

Runtime: 54074.39 s.
